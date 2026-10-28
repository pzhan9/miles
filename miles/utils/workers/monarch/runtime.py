"""Monarch as the runtime of the worker manager: one Monarch ``ProcMesh`` per cell.

Monarch neither assigns gpus nor knows Miles' slots, so the layout here maps every logical gpu slot onto a
``(host, device)`` of the worker host mesh, packing hosts in mesh order like a Ray ``PACK`` placement group.
Every gpu worker sees its whole host, as under Ray, and selects the devices the manager placed it on.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from monarch.actor import Actor, HostMesh, ProcMesh, current_rank, default_bootstrap_cmd, endpoint

from miles.utils.misc import get_current_node_ip
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.worker_manager.runtime import GpuSlotRef, WorkerLaunch, WorkerProcess, WorkerRuntime

logger = logging.getLogger(__name__)

_PROBE_IS_DEAD_TIMEOUT_SECONDS = 1.0
_STOP_TIMEOUT_SECONDS = 60.0
_PROCS_DIM = "procs"


@dataclass(frozen=True, kw_only=True)
class GpuSlot:
    host_rank: int
    host_ip: str
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


def compute_monarch_gpu_layout(
    *, request: GpuLayoutRequest, host_ips: list[str], num_gpus_per_host: int
) -> MonarchGpuLayout:
    num_gpus = request.num_gpus
    capacity = len(host_ips) * num_gpus_per_host
    assert num_gpus <= capacity, (
        f"the run lays its gpu workers out on {num_gpus} gpus, but the {len(host_ips)} monarch hosts offer "
        f"{num_gpus_per_host} each ({capacity} in total)"
    )
    slots = [
        GpuSlot(host_rank=host_rank, host_ip=host_ip, gpu_id=gpu_id)
        for host_rank, host_ip in enumerate(host_ips)
        for gpu_id in range(num_gpus_per_host)
    ][:num_gpus]
    slots_of_layout = {"actor": slots, "rollout": slots[request.rollout_offset :]}
    if request.use_critic:
        slots_of_layout["critic"] = slots_of_layout["actor"]
    return MonarchGpuLayout(slots_of_layout=slots_of_layout)


class MonarchNodeProbe(Actor):
    @endpoint
    def probe(self) -> tuple[str, int]:
        """(address peers reach this host on, gpus visible to processes the host spawns)."""
        return get_current_node_ip(), _count_visible_gpus()


def _count_visible_gpus() -> int:
    if (cvd := os.environ.get("CUDA_VISIBLE_DEVICES")) is not None:
        return len([device for device in cvd.split(",") if device.strip()])
    return len([name for name in os.listdir("/dev") if name.startswith("nvidia") and name[len("nvidia") :].isdigit()])


async def probe_hosts(worker_hosts: HostMesh) -> tuple[list[str], int]:
    """Every host's address in mesh order, and the gpus the least endowed host offers."""
    probe_procs = worker_hosts.spawn_procs(per_host={_PROCS_DIM: 1}, name="miles-node-probe")
    try:
        probes = probe_procs.spawn("miles-node-probe", MonarchNodeProbe)
        answers = await probes.probe.call()
        results = [answer for _, answer in sorted(answers.items(), key=lambda item: item[0].rank)]
    finally:
        await probe_procs.stop()
    return [ip for ip, _ in results], min(num_gpus for _, num_gpus in results)


class MonarchWorkerActor(Actor):
    """Hosts one launched worker of a cell; the cell's proc mesh hosts one of these per worker.

    Every rank receives the targets of the whole cell and builds the one at its own rank, so a cell is spawned
    with a single ``ProcMesh.spawn``.
    """

    def __init__(self, targets: list[tuple[type, dict[str, Any]]], manager: Any) -> None:
        install_manager_handle(manager)
        actor_class, ctor_kwargs = targets[current_rank().rank]
        self._target = actor_class(**ctor_kwargs)

    @endpoint
    def invoke(self, method: str, kwargs: dict[str, Any]) -> Any:
        return getattr(self._target, method)(**kwargs)  # config-access-exempt: attribute selected at runtime

    @endpoint
    def ping(self) -> bool:
        return True


# The worker manager of this run, as seen from this process: the driver installs it when it launches the manager,
# and every launched worker when it starts, since Monarch has no registry to look an actor up by name.
_MANAGER_HANDLE: Any = None


def install_manager_handle(handle: Any) -> None:
    global _MANAGER_HANDLE
    _MANAGER_HANDLE = handle


def installed_manager_handle() -> Any:
    assert _MANAGER_HANDLE is not None, "this process was not launched by a monarch worker manager"
    return _MANAGER_HANDLE


@dataclass(eq=False)
class _CellMesh:
    cell_id: str
    proc_mesh: ProcMesh
    actor_mesh: Any
    stopped: bool = False


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
        self._live_meshes: dict[str, _CellMesh] = {}

    async def launch_cell(self, *, cell_id: str, launches: list[WorkerLaunch]) -> list[WorkerProcess]:
        host_mesh, host_ranks = self._place(launches)
        num_hosts = len(set(host_ranks))
        procs_per_host = len(launches) // num_hosts
        mesh_name = f"{cell_id}-{next(self._mesh_counter)}"
        host_dim = _host_dim_of(host_mesh)
        host_slice = (
            host_mesh if host_dim is None else host_mesh.slice(**{host_dim: slice(host_ranks[0], host_ranks[-1] + 1)})
        )

        base_command = default_bootstrap_cmd()
        envs = [self._compute_env(launch) for launch in launches]
        proc_mesh = host_slice.spawn_procs(
            per_host={_PROCS_DIM: procs_per_host},
            name=mesh_name,
            bootstrap_command=lambda point: base_command.with_env(envs[point.rank]),
        )
        actor_mesh = proc_mesh.spawn(
            mesh_name,
            MonarchWorkerActor,
            [(launch.actor_class, launch.ctor_kwargs) for launch in launches],
            self._manager,
        )
        cell_mesh = _CellMesh(cell_id=cell_id, proc_mesh=proc_mesh, actor_mesh=actor_mesh)
        self._live_meshes[mesh_name] = cell_mesh
        await actor_mesh.ping.call()
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

    async def release_cell(self, *, cell_id: str, processes: list[WorkerProcess]) -> None:
        cell_meshes = {process.cell_mesh for process in processes if isinstance(process, _MonarchWorkerProcess)}
        for cell_mesh in cell_meshes:
            self._live_meshes = {name: mesh for name, mesh in self._live_meshes.items() if mesh is not cell_mesh}
            cell_mesh.stopped = True
            try:
                await asyncio.wait_for(cell_mesh.proc_mesh.stop(), timeout=_STOP_TIMEOUT_SECONDS)
                logger.info(f"Stopped the proc mesh of cell {cell_id}")
            except Exception as e:
                logger.warning(f"Failed to stop the proc mesh of cell {cell_id} ({e})")

    def physical_gpu_ids(self, slots: GpuSlotRef) -> list[int]:
        layout_slots = self._layout.slots_of_layout[slots.layout_name]
        return [layout_slots[slots.first_slot_index + i].gpu_id for i in range(slots.num_slots)]

    def worker_capability_factory(self) -> Callable[[], BackendCapability]:
        return self._capability_factory

    def cell_id_of_failed_mesh(self, mesh_name: str) -> str | None:
        cell_mesh = self._live_meshes.get(mesh_name)
        return None if cell_mesh is None else cell_mesh.cell_id

    def _place(self, launches: list[WorkerLaunch]) -> tuple[HostMesh, list[int]]:
        gpu_slots = [launch.gpu_slots for launch in launches]
        if all(slots is None for slots in gpu_slots):
            return self._head_host, [0] * len(launches)
        assert all(slots is not None for slots in gpu_slots), f"a cell mixes gpu and cpu-only workers: {launches}"

        host_ranks = [
            self._layout.slots_of_layout[slots.layout_name][slots.first_slot_index].host_rank for slots in gpu_slots
        ]
        counts = [len(list(group)) for _, group in itertools.groupby(host_ranks)]
        assert host_ranks == sorted(host_ranks) and len(set(counts)) == 1 and len(counts) == len(set(host_ranks)), (
            f"monarch spawns a cell as one proc mesh, which needs the same number of workers on each of a run of "
            f"consecutive hosts, but the cell's workers sit on hosts {host_ranks}"
        )
        assert host_ranks[-1] - host_ranks[0] + 1 == len(counts), f"the cell skips hosts: {host_ranks}"
        return self._worker_hosts, host_ranks

    @staticmethod
    def _compute_env(launch: WorkerLaunch) -> dict[str, str | None]:
        # Ray hides every gpu from a worker it gives none; a gpu worker keeps the whole host it inherits.
        hidden_gpus = {"CUDA_VISIBLE_DEVICES": ""} if launch.gpu_slots is None else {}
        return {**launch.env_vars, **hidden_gpus}


def _host_dim_of(host_mesh: HostMesh) -> str | None:
    """The dimension a host mesh lays its hosts along; the driver's own host is a mesh with none."""
    labels = list(host_mesh.extent.keys())
    assert len(labels) <= 1, f"a worker host mesh is laid out along one dimension, not {labels}"
    return labels[0] if labels else None


class _MonarchWorkerProcess(WorkerProcess):
    def __init__(self, *, cell_mesh: _CellMesh, actor: Any) -> None:
        self.cell_mesh = cell_mesh
        self._actor = actor

    async def call(self, method: str, /, **kwargs: Any) -> Any:
        return await self._actor.invoke.call_one(method, kwargs)

    def call_without_result(self, method: str, /, **kwargs: Any) -> None:
        self._actor.invoke.broadcast(method, kwargs)

    async def probe_is_dead(self) -> bool:
        if self.cell_mesh.stopped:
            return True
        try:
            await asyncio.wait_for(self._actor.ping.call_one(), timeout=_PROBE_IS_DEAD_TIMEOUT_SECONDS)
        except (TimeoutError, asyncio.TimeoutError):
            return False
        except Exception as e:
            logger.info(f"Worker of cell {self.cell_mesh.cell_id} failed its liveness probe: {e!r}")
            return True
        return False

    @property
    def native_handle(self) -> Any:
        return self._actor


def make_capability_factory(create: Callable[[Any], BackendCapability]) -> Callable[[], BackendCapability]:
    return partial(_create_from_installed_manager, create=create)


def _create_from_installed_manager(*, create: Callable[[Any], BackendCapability]) -> BackendCapability:
    return create(installed_manager_handle())
