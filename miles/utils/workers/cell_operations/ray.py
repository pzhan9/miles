from __future__ import annotations

import ray.actor

from miles.utils.workers.worker_manager.client import ManagerCellOperations
from miles.utils.workers.worker_provider.ray import RayWorkerManagerClient


class RayCellOperations(ManagerCellOperations):
    def __init__(self, *, worker_manager_handle: ray.actor.ActorHandle) -> None:
        super().__init__(RayWorkerManagerClient(worker_manager_handle))
