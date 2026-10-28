from __future__ import annotations

import abc
import asyncio
from typing import Any

from miles.utils.workers.worker_handle import BaseWorkerHandle


class BaseCellHandle(abc.ABC):
    """Calls the same method on every worker of one cell and answers their results in worker order."""

    @abc.abstractmethod
    async def call_each(self, method: str, /, kwargs_per_worker: list[dict[str, Any]]) -> list[Any]: ...


class GatherCellHandle(BaseCellHandle):
    """One call per worker handle, gathered: what every transport without a mesh call does."""

    def __init__(self, worker_handles: list[BaseWorkerHandle]) -> None:
        self._worker_handles = worker_handles

    async def call_each(self, method: str, /, kwargs_per_worker: list[dict[str, Any]]) -> list[Any]:
        assert len(kwargs_per_worker) == len(
            self._worker_handles
        ), f"{len(kwargs_per_worker)} sets of arguments for the {len(self._worker_handles)} workers of a cell"
        return list(
            await asyncio.gather(
                *(
                    getattr(handle, method)(**kwargs)  # config-access-exempt: attribute selected at runtime
                    for handle, kwargs in zip(self._worker_handles, kwargs_per_worker, strict=True)
                )
            )
        )
