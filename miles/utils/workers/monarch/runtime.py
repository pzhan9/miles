"""Monarch as the runtime of the worker manager: one Monarch ``ProcMesh`` per cell.

Monarch neither assigns gpus nor knows Miles' slots, so the layout here maps every logical gpu slot onto a
``(host, device)`` of the worker host mesh, packing hosts in mesh order like a Ray ``PACK`` placement group.
Every gpu worker sees its whole host, as under Ray, and selects the devices the manager placed it on.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import logging
import os
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any

from monarch.actor import Actor, HostMesh, ProcMesh, concurrent_endpoint, current_rank, default_bootstrap_cmd, endpoint

from miles.utils.misc import VISIBLE_DEVICES_ENV_VARS, get_current_node_ip, visible_devices
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.env_vars import WORKER_GPU_IDS_ENV_VAR
from miles.utils.workers.monarch.handles import call_endpoint, probe_cell_is_dead
from miles.utils.workers.worker_handle import WorkerUnreachableError
from miles.utils.workers.worker_manager.runtime import GpuSlotRef, WorkerLaunch, WorkerProcess, WorkerRuntime

logger = logging.getLogger(__name__)

_STOP_TIMEOUT_SECONDS = 60.0
_PROCS_DIM = "procs"


@dataclass(frozen=True, kw_only=True)
class GpuSlot:
    host_rank: int
    gpu_id: int


@dataclass(frozen=True, kw_only=True)
class GpuLayoutRequest:
    """What the driver asks of the gpu layout, as it asks Ray for placement groups."""

    num_gpus: int
    rollout_offset: int
    use_critic: bool
    num_gpus_per_node: int


@dataclass(frozen=True, kw_only=True)
class MonarchGpuLayout:
    """The same role as Ray's placement groups: logical slots of each named layout, in placement order."""

    slots_of_layout: dict[str, list[GpuSlot]]


@dataclass(frozen=True, kw_only=True)
class ProbedHost:
    ip: str
    # physical ids, in the order the host's processes see them; what to_local_gpu_id maps a worker's gpu ids from
    gpu_ids: list[int]


def compute_monarch_gpu_layout(*, request: GpuLayoutRequest, hosts: list[ProbedHost]) -> MonarchGpuLayout:
    num_gpus = request.num_gpus
    num_gpus_per_host = min([len(host.gpu_ids) for host in hosts] + [request.num_gpus_per_node])
    capacity = len(hosts) * num_gpus_per_host
    assert num_gpus <= capacity, (
        f"the run lays its gpu workers out on {num_gpus} gpus, but the {len(hosts)} monarch hosts offer "
        f"{num_gpus_per_host} each ({capacity} in total)"
    )
    slots = [
        GpuSlot(host_rank=host_rank, gpu_id=gpu_id)
        for host_rank, host in enumerate(hosts)
        for gpu_id in host.gpu_ids[:num_gpus_per_host]
    ][:num_gpus]
    slots_of_layout = {"actor": slots, "rollout": slots[request.rollout_offset :]}
    if request.use_critic:
        slots_of_layout["critic"] = slots_of_layout["actor"]
    return MonarchGpuLayout(slots_of_layout=slots_of_layout)


class MonarchNodeProbe(Actor):
    @endpoint
    def probe(self) -> ProbedHost:
        # TODO: get_current_node_ip asks Ray's private lookup, which off a Ray node probes the default route;
        # Monarch could offer its own node address instead, but that lookup already works, so no need yet.
        return ProbedHost(ip=get_current_node_ip(), gpu_ids=visible_gpu_ids())


def visible_gpu_ids() -> list[int]:
    """The physical ids of the gpus processes on this host can use."""
    if (visible := visible_devices()) is not None:
        devices = [device.strip() for device in visible.split(",") if device.strip()]
        assert all(device.isdigit() for device in devices), f"monarch places gpus by index, not by uuid: {visible}"
        return [int(device) for device in devices]
    if any(name in os.environ for name in VISIBLE_DEVICES_ENV_VARS):
        # set, but to an empty list: every gpu is hidden
        return []
    devices = [name[len("nvidia") :] for name in os.listdir("/dev") if name.startswith("nvidia")]
    return sorted(int(index) for index in devices if index.isdigit())


async def probe_hosts(worker_hosts: HostMesh) -> list[ProbedHost]:
    """Every host's address and visible gpus, in mesh order."""
    probe_procs = worker_hosts.spawn_procs(per_host={_PROCS_DIM: 1}, name="miles-node-probe")
    try:
        probes = probe_procs.spawn("miles-node-probe", MonarchNodeProbe)
        answers = await probes.probe.call()
    finally:
        await probe_procs.stop()
    return [host for _, host in sorted(answers.items(), key=lambda item: item[0].rank)]


class MonarchWorkerActor(Actor):
    """Hosts one launched worker of a cell; the cell's proc mesh hosts one of these per worker.

    Every rank receives the targets of the whole cell and builds the one at its own rank, so a cell is spawned
    with a single ``ProcMesh.spawn``.
    """

    def __init__(self, launches: list[WorkerLaunch], manager: Any) -> None:
        install_manager_handle(manager)
        target = launches[current_rank().rank]
        self._target = target.actor_class(**target.ctor_kwargs)
        # the launcher's own verbs (node probes, run, shutdown, ...) run on one thread of their own
        self._launcher_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="monarch-launcher")

    # Every call runs as a task of its own, so one still running blocks no liveness ping.
    @concurrent_endpoint
    async def invoke(self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        return await self._dispatch(method, args=args, kwargs=kwargs)

    @concurrent_endpoint
    async def ping(self) -> bool:
        return True

    async def _dispatch(self, method: str, *, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        fn = getattr(self._target, method)  # config-access-exempt: attribute selected at runtime from method
        if inspect.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        return await asyncio.get_running_loop().run_in_executor(self._launcher_thread, partial(fn, *args, **kwargs))


# The worker manager of this run, as seen from this process: the driver installs it when it launches the manager,
# and every launched worker when it starts, since Monarch has no registry to look an actor up by name.
_MANAGER_HANDLE: Any = None


def install_manager_handle(handle: Any) -> None:
    global _MANAGER_HANDLE
    _MANAGER_HANDLE = handle


def installed_manager_handle() -> Any:
    assert _MANAGER_HANDLE is not None, "this process was not launched by a monarch worker manager"
    return _MANAGER_HANDLE


@dataclass(frozen=True, kw_only=True)
class _CellMesh:
    name: str
    proc_mesh: ProcMesh
    actor_mesh: Any


class MonarchWorkerRuntime(WorkerRuntime):
    def __init__(
        self,
        *,
        head_host: HostMesh,
        worker_hosts: HostMesh,
        layout: MonarchGpuLayout,
        manager: Any,
        capability_factory: Callable[[], BackendCapability],
    ) -> None:
        self._head_host = head_host
        self._worker_hosts = worker_hosts
        self._layout = layout
        self._manager = manager
        self._capability_factory = capability_factory
        self._mesh_counter = itertools.count()

    async def launch_cell(self, *, cell_id: str, launches: list[WorkerLaunch]) -> list[WorkerProcess]:
        host_mesh, host_ranks, procs_per_host = self._place(launches)
        mesh_name = f"{cell_id}-{next(self._mesh_counter)}"
        host_dim = _host_dim_of(host_mesh)
        host_slice = (
            host_mesh if host_dim is None else host_mesh.slice(**{host_dim: slice(host_ranks[0], host_ranks[-1] + 1)})
        )

        base_command = default_bootstrap_cmd()
        envs = [
            _compute_env(
                launch,
                gpu_ids=None if launch.gpu_slots is None else self.physical_gpu_ids(launch.gpu_slots),
            )
            for launch in launches
        ]
        proc_mesh = host_slice.spawn_procs(
            per_host={_PROCS_DIM: procs_per_host},
            name=mesh_name,
            bootstrap_command=lambda point: base_command.with_env(envs[point.rank]),
        )
        try:
            actor_mesh = proc_mesh.spawn(mesh_name, MonarchWorkerActor, launches, self._manager)
            await actor_mesh.ping.call()
        except BaseException:
            # The core never learns of processes it was not handed, so nobody else would stop this mesh.
            await _stop_proc_mesh(proc_mesh, mesh_name=mesh_name)
            raise
        cell_mesh = _CellMesh(name=mesh_name, proc_mesh=proc_mesh, actor_mesh=actor_mesh)
        return [
            _MonarchWorkerProcess(
                cell_mesh=cell_mesh,
                actor=actor_mesh.slice(
                    **({} if host_dim is None else {host_dim: index // procs_per_host}),
                    **{_PROCS_DIM: index % procs_per_host},
                ),
            )
            for index in range(len(launches))
        ]

    async def kill_worker(self, process: WorkerProcess) -> None:
        # A worker lives in its cell's proc mesh, which only stops as a whole in release_cell.
        pass

    async def release_cell(self, *, processes: list[WorkerProcess]) -> None:
        # every worker of a cell lives in the one mesh its launch_cell spawned
        [cell_mesh] = {process.cell_mesh for process in processes}
        await _stop_proc_mesh(cell_mesh.proc_mesh, mesh_name=cell_mesh.name)

    def physical_gpu_ids(self, slots: GpuSlotRef) -> list[int]:
        layout_slots = self._layout.slots_of_layout[slots.layout_name]
        return [layout_slots[slots.first_slot_index + i].gpu_id for i in range(slots.num_slots)]

    def worker_capability_factory(self) -> Callable[[], BackendCapability]:
        return self._capability_factory

    def _place(self, launches: list[WorkerLaunch]) -> tuple[HostMesh, list[int], int]:
        gpu_slots = [launch.gpu_slots for launch in launches]
        if all(slots is None for slots in gpu_slots):
            return self._head_host, [0] * len(launches), len(launches)
        assert all(slots is not None for slots in gpu_slots), f"a cell mixes gpu and cpu-only workers: {launches}"

        host_ranks = [
            self._layout.slots_of_layout[slots.layout_name][slots.first_slot_index].host_rank for slots in gpu_slots
        ]
        procs_per_host = len(host_ranks) // (host_ranks[-1] - host_ranks[0] + 1)
        assert procs_per_host and host_ranks == [
            host_ranks[0] + i // procs_per_host for i in range(len(host_ranks))
        ], (
            f"monarch spawns a cell as one proc mesh, which needs the same number of workers on each of a run of "
            f"consecutive hosts, but the cell's workers sit on hosts {host_ranks}"
        )
        return self._worker_hosts, host_ranks, procs_per_host


def _compute_env(launch: WorkerLaunch, *, gpu_ids: list[int] | None) -> dict[str, str | None]:
    if gpu_ids is None:
        # Ray hides every gpu from a worker it gives none
        gpus = dict.fromkeys(VISIBLE_DEVICES_ENV_VARS, "")
    else:
        # a gpu worker keeps the whole host it inherits, as under Ray, and learns which devices are its own
        gpus = {WORKER_GPU_IDS_ENV_VAR: ",".join(str(gpu_id) for gpu_id in gpu_ids)}
    return {**launch.env_vars, **gpus}


async def _stop_proc_mesh(proc_mesh: ProcMesh, *, mesh_name: str) -> None:
    try:
        await asyncio.wait_for(proc_mesh.stop(), timeout=_STOP_TIMEOUT_SECONDS)
        logger.info(f"Stopped proc mesh {mesh_name}")
    except Exception as e:
        logger.warning(f"Failed to stop proc mesh {mesh_name} ({e})")


def _host_dim_of(host_mesh: HostMesh) -> str | None:
    """The dimension a host mesh lays its hosts along; the driver's own host is a mesh with none."""
    labels = list(host_mesh.extent.keys())
    assert len(labels) <= 1, f"a worker host mesh is laid out along one dimension, not {labels}"
    return labels[0] if labels else None


class _MonarchWorkerProcess(WorkerProcess):
    def __init__(self, *, cell_mesh: _CellMesh, actor: Any) -> None:
        self.cell_mesh = cell_mesh
        self._actor = actor
        self._unawaited_calls: set[asyncio.Task] = set()

    async def call(self, method: str, /, **kwargs: Any) -> Any:
        return await call_endpoint(self._actor.invoke.call_one(method, (), kwargs), describe=method)

    def call_without_result(self, method: str, /, **kwargs: Any) -> None:
        # Nobody awaits this call, so its failure is logged here, as ray logs the error of a call nobody awaits.
        task = asyncio.get_running_loop().create_task(_log_if_failed(self.call(method, **kwargs), describe=method))
        self._unawaited_calls.add(task)
        task.add_done_callback(self._unawaited_calls.discard)

    async def probe_is_dead(self) -> bool:
        return await probe_cell_is_dead(self.cell_mesh.actor_mesh)

    @property
    def native_handle(self) -> Any:
        return self._actor


async def _log_if_failed(call: Awaitable[Any], *, describe: str) -> None:
    try:
        await call
    except WorkerUnreachableError:
        # the worker is gone, which the liveness scan reports for its whole cell
        pass
    except Exception:
        logger.error(f"{describe!r}, which nobody awaits, failed on its worker", exc_info=True)
