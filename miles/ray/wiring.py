from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial

import ray
from ray.actor import ActorHandle

from miles.ray.gpu_layout import compute_gpu_layout
from miles.ray.specs.entrypoint import compute_specs
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.args.runtime import AllConfig
from miles.utils.workers.backend_capability import factory
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.connection_config import build_static_conn_config
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.types import ClusterBackend, WorkerCommBackend


@dataclass(frozen=True)
class LaunchedMonarchWorkerManager:
    shutdown: Callable[[], Awaitable[None]]


def launch_worker_manager(args):
    match ClusterBackend(args.cluster_backend):
        case ClusterBackend.KUBERNETES:
            return None
        case ClusterBackend.RAY:
            return _launch_ray_worker_manager(args)
        case ClusterBackend.MONARCH:
            return _launch_monarch_worker_manager(args)


async def shutdown_worker_manager(worker_manager_handle: ActorHandle | LaunchedMonarchWorkerManager | None) -> None:
    if worker_manager_handle is None:
        return
    if isinstance(worker_manager_handle, LaunchedMonarchWorkerManager):
        await worker_manager_handle.shutdown()
        return
    try:
        await worker_manager_handle.shutdown.remote()
    finally:
        ray.kill(worker_manager_handle)


def compute_backend_capability(args: AllConfig) -> BackendCapability:
    return factory.get_backend_capability(
        static_connections=build_static_conn_config(specs=compute_specs(args), scaling=ScalingConfig.slice_from(args)),
        cluster_backend=ClusterBackend(args.cluster_backend),
    )


def _launch_ray_worker_manager(args):
    from miles.ray.placement_group import create_placement_groups

    specs = compute_specs(args)
    # TODO: pass in specs instead of args
    pgs = create_placement_groups(args)
    return RayWorkerManager.launch(args, specs, pgs, comm_backend=WorkerCommBackend(args.worker_comm_backend))


def _launch_monarch_worker_manager(args) -> LaunchedMonarchWorkerManager:
    # torchmonarch is an optional dependency, imported only by runs that pick it
    from miles.utils.workers.monarch.manager import MonarchWorkerManager, shutdown_monarch_worker_manager
    from miles.utils.workers.monarch.runtime import GpuLayoutRequest

    num_gpus, rollout_offset = compute_gpu_layout(args)
    manager = MonarchWorkerManager.launch(
        args,
        compute_specs(args),
        comm_backend=WorkerCommBackend(args.worker_comm_backend),
        gpu_layout=GpuLayoutRequest(
            num_gpus=num_gpus,
            rollout_offset=rollout_offset,
            use_critic=args.use_critic,
            num_gpus_per_node=args.num_gpus_per_node,
        ),
    )
    return LaunchedMonarchWorkerManager(shutdown=partial(shutdown_monarch_worker_manager, manager))
