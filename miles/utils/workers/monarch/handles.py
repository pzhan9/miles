"""Calling workers through Monarch endpoints rather than rpc: ``--worker-comm-backend monarch``.

A worker handle reaches one worker; a cell handle reaches every worker of a cell with a single mesh call, which
Monarch fans out down its own routing tree and answers as one ``ValueMesh``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from monarch._rust_bindings.monarch_hyperactor.supervision import SupervisionError
from monarch.actor import ActorError

from miles.utils.workers.cell_handle import BaseCellHandle
from miles.utils.workers.worker_handle import (
    _WAIT_DEAD_PROBE_INTERVAL_SECONDS,
    BaseWorkerHandle,
    WorkerUnreachableError,
)


class MonarchWorkerHandle(BaseWorkerHandle):
    def __init__(self, *, actor: Any, method_names: frozenset[str]) -> None:
        self._actor = actor
        self._method_names = method_names

    def __getattr__(self, name: str) -> Callable[..., Awaitable[Any]]:
        if name.startswith("_") or name not in self._method_names:
            raise AttributeError(name)

        async def call(*args: Any, **kwargs: Any) -> Any:
            return await call_endpoint(self._actor.invoke.call_one(name, args, kwargs), describe=name)

        return call

    async def wait_ready(self, *, timeout: float, allow_server_uuid_change: bool = False) -> None:
        del allow_server_uuid_change
        try:
            await asyncio.wait_for(call_endpoint(self._actor.ping.call_one(), describe="ping"), timeout=timeout)
        except (TimeoutError, asyncio.TimeoutError) as e:
            raise WorkerUnreachableError(f"Worker not ready within {timeout}s") from e

    async def wait_idle(self, *, timeout: float) -> None:
        raise NotImplementedError(
            "a monarch actor does not track the calls it is running, so nobody can wait for it to go idle; only the "
            "rpc communication backend answers this"
        )

    async def probe_is_dead(self) -> bool:
        try:
            await asyncio.wait_for(self._actor.ping.call_one(), timeout=_WAIT_DEAD_PROBE_INTERVAL_SECONDS)
        except (TimeoutError, asyncio.TimeoutError):
            return False
        except (SupervisionError, ActorError):
            return True
        return False


class MonarchCellHandle(BaseCellHandle):
    def __init__(self, *, actor_mesh: Any, num_workers: int) -> None:
        self._actor_mesh = actor_mesh
        self._num_workers = num_workers

    async def call_each(self, method: str, /, kwargs_per_worker: list[dict[str, Any]]) -> list[Any]:
        assert (
            len(kwargs_per_worker) == self._num_workers
        ), f"{len(kwargs_per_worker)} sets of arguments for the {self._num_workers} workers of a cell"
        results = await call_endpoint(self._actor_mesh.invoke_each.call(method, kwargs_per_worker), describe=method)
        return [value for _, value in sorted(results.items(), key=lambda item: item[0].rank)]


def get_endpoint_result(future: Any, *, describe: str) -> Any:
    """The blocking form of ``call_endpoint``, for code that runs outside any event loop."""
    try:
        return future.get()
    except ActorError as e:
        raise e.exception from e
    except SupervisionError as e:
        raise WorkerUnreachableError(f"Worker died or is unreachable when calling {describe!r}: {e!r}") from e


async def call_endpoint(future: Any, *, describe: str) -> Any:
    """Answer what the call answered, raising the worker's own exception the way a Ray actor call does."""
    try:
        return await future
    except ActorError as e:
        raise e.exception from e
    except SupervisionError as e:
        raise WorkerUnreachableError(f"Worker died or is unreachable when calling {describe!r}: {e!r}") from e
