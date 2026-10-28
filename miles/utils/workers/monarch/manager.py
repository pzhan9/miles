"""The worker manager of a ``--cluster-backend monarch`` run, as a Monarch actor that owns every cell's proc mesh."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from monarch.actor import Actor, HostMesh, MeshFailure, endpoint, this_host

from miles.utils.test_utils.fault_injector.controller import FaultHookCommand
from miles.utils.test_utils.fault_injector.models import FaultHookRecord, ObservedFaultHookTarget
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.cell_handle import BaseCellHandle
from miles.utils.workers.monarch.hosts import acquire_worker_hosts
from miles.utils.workers.monarch.runtime import (
    GpuLayoutRequest,
    MonarchWorkerRuntime,
    compute_monarch_gpu_layout,
    install_manager_handle,
    make_capability_factory,
    probe_hosts,
)
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_manager.client import ManagerBackendCapability, WorkerManagerClient
from miles.utils.workers.worker_manager.core import WorkerManager
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import BaseSpec, NamedHostAndPorts

logger = logging.getLogger(__name__)

_MANAGER_NAME = "miles-worker-manager"


class MonarchWorkerManager(Actor):
    def __init__(self) -> None:
        self._core = WorkerManager()
        self._runtime: MonarchWorkerRuntime | None = None

    @staticmethod
    def launch(args, specs: list[BaseSpec], *, comm_backend: WorkerCommBackend, gpu_layout: GpuLayoutRequest) -> Any:
        """Spawn the manager in a process of its own on the driver's host, and start every cell."""
        head_host = this_host()
        worker_hosts = acquire_worker_hosts(head_host=head_host)
        manager_procs = head_host.spawn_procs(per_host={"procs": 1}, name=_MANAGER_NAME)
        manager = manager_procs.spawn(_MANAGER_NAME, MonarchWorkerManager)
        install_manager_handle(manager)
        manager.init.call_one(
            args,
            specs,
            comm_backend=comm_backend,
            gpu_layout=gpu_layout,
            head_host=head_host,
            worker_hosts=worker_hosts,
            manager=manager,
        ).get()
        return manager

    @endpoint
    async def init(
        self,
        args,
        specs: list[BaseSpec],
        *,
        comm_backend: WorkerCommBackend,
        gpu_layout: GpuLayoutRequest,
        head_host: HostMesh,
        worker_hosts: HostMesh,
        manager: Any,
    ) -> None:
        install_manager_handle(manager)
        host_ips, num_gpus_per_host = await probe_hosts(worker_hosts)
        layout = compute_monarch_gpu_layout(
            request=gpu_layout,
            host_ips=host_ips,
            num_gpus_per_host=min(num_gpus_per_host, gpu_layout.num_gpus_per_node),
        )
        logger.info(f"Monarch hosts {host_ips} lay out {gpu_layout.num_gpus} gpus as {layout}")
        self._runtime = MonarchWorkerRuntime(
            head_host=head_host,
            worker_hosts=worker_hosts,
            layout=layout,
            manager=manager,
            capability_factory=make_capability_factory(create_monarch_backend_capability),
        )
        await self._core.init_pools(args, specs, runtime=self._runtime, comm_backend=comm_backend)

    @endpoint
    async def get_cell_infos(self, *, pool_ids: list[str] | None, category: str | None) -> dict[str, CellInfo]:
        return self._core.get_cell_infos(pool_ids=pool_ids, category=category)

    @endpoint
    async def get_worker_infos(self, cell_id: str) -> list[WorkerInfo]:
        return self._core.get_worker_infos(cell_id)

    @endpoint
    async def get_worker_addrs(self, worker_name: str) -> NamedHostAndPorts:
        return self._core.get_worker_addrs(worker_name)

    @endpoint
    async def get_actor_handle(self, worker_name: str, *, expected_generation: int) -> Any:
        return self._core.get_actor_handle(worker_name, expected_generation=expected_generation)

    @endpoint
    async def get_cell_native_handle(self, cell_id: str, *, expected_generation: int) -> Any | None:
        return self._core.get_cell_native_handle(cell_id, expected_generation=expected_generation)

    @endpoint
    async def start_cells(self, cell_ids: list[str]) -> None:
        await self._core.start_cells(cell_ids)

    @endpoint
    async def stop_cells(self, cell_ids: list[str]) -> None:
        await self._core.stop_cells(cell_ids)

    @endpoint
    async def shutdown(self) -> None:
        await self._core.shutdown()

    @endpoint
    async def observe_fault_target(self, cell_id: str, *, rank: int) -> ObservedFaultHookTarget:
        return self._core.observe_fault_target(cell_id, rank=rank)

    @endpoint
    async def control_fault_hook(self, command: FaultHookCommand) -> FaultHookRecord:
        return await self._core.control_fault_hook(command)

    async def __supervise__(self, failure: MeshFailure) -> bool:
        """A worker process died: tear its cell down as the liveness scan would, and keep the run alive."""
        cell_id = None if self._runtime is None else self._runtime.cell_id_of_failed_mesh(failure.mesh_name)
        if cell_id is None:
            logger.info(f"Ignoring a failure of mesh {failure.mesh_name}, which no live cell owns: {failure.report()}")
            return True
        logger.error(f"Cell {cell_id} lost a worker: {failure.report()}")
        generation = self._core._find_cell(cell_id).generation
        asyncio.create_task(
            self._core.report_lost_cell(cell_id, generation=generation, reason=f"lost a worker ({failure.mesh_name})")
        )
        return True


class MonarchWorkerManagerClient(WorkerManagerClient):
    def __init__(self, manager: Any) -> None:
        self._manager = manager

    async def get_cell_infos(self, *, pool_ids: list[str] | None, category: str | None) -> dict[str, CellInfo]:
        return await self._manager.get_cell_infos.call_one(pool_ids=pool_ids, category=category)

    async def get_worker_addrs(self, worker_name: str) -> NamedHostAndPorts:
        return await self._manager.get_worker_addrs.call_one(worker_name)

    async def start_cells(self, cell_ids: list[str]) -> None:
        await self._manager.start_cells.call_one(cell_ids)

    async def stop_cells(self, cell_ids: list[str]) -> None:
        await self._manager.stop_cells.call_one(cell_ids)

    async def observe_fault_target(self, cell_id: str, *, rank: int) -> ObservedFaultHookTarget:
        return await self._manager.observe_fault_target.call_one(cell_id, rank=rank)

    async def control_fault_hook(self, command: FaultHookCommand) -> FaultHookRecord:
        return await self._manager.control_fault_hook.call_one(command)

    def get_worker_infos_blocking(self, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [self._manager.get_worker_infos.call_one(cell_id).get() for cell_id in cell_ids]

    def get_native_handles_blocking(self, infos: list[WorkerInfo]) -> dict[str, BaseWorkerHandle]:
        handles = {
            info.name: self._manager.get_actor_handle.call_one(info.name, expected_generation=info.generation).get()
            for info in infos
        }
        not_served = sorted(name for name, handle in handles.items() if not isinstance(handle, BaseWorkerHandle))
        assert not not_served, f"{not_served} are launched as commands, which have no methods to call"
        return handles

    def get_native_cell_handle_blocking(self, cell_id: str, *, generation: int) -> BaseCellHandle | None:
        return self._manager.get_cell_native_handle.call_one(cell_id, expected_generation=generation).get()


def create_monarch_backend_capability(manager: Any) -> BackendCapability:
    return ManagerBackendCapability(MonarchWorkerManagerClient(manager))


async def shutdown_monarch_worker_manager(manager: Any) -> None:
    await manager.shutdown.call_one()
