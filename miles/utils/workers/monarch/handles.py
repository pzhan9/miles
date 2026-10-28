"""Calling workers through Monarch endpoints rather than rpc: ``--worker-comm-backend monarch``.

A worker handle reaches one worker; a cell handle reaches every worker of a cell with a single mesh call, which
Monarch fans out down its own routing tree and answers as one ``ValueMesh``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

# TODO: import SupervisionError from a public Monarch module once Monarch exports it
from monarch._rust_bindings.monarch_hyperactor.supervision import SupervisionError
from monarch.actor import ActorError

from miles.utils.workers.cell_handle import BaseCellHandle
from miles.utils.workers.worker_handle import (
    _WAIT_DEAD_PROBE_INTERVAL_SECONDS,
    BaseWorkerHandle,
    WorkerUnreachableError,
)


class MonarchWorkerHandle(BaseWorkerHandle):
    def __init__(self, *, actor: Any, cell: MonarchCellHandle, rank: int, method_names: frozenset[str]) -> None:
        self._actor = actor
        self._cell = cell
        self._rank = rank
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
        return await probe_cell_is_dead(self._cell.actor_mesh)

    def cell_handle_with(self, cell_worker_handles: list[BaseWorkerHandle]) -> BaseCellHandle | None:
        # one mesh call reaches the workers only if they are exactly the ranks of this one's mesh, in rank order
        ranks = [
            handle._rank if isinstance(handle, MonarchWorkerHandle) and handle._cell.name == self._cell.name else None
            for handle in cell_worker_handles
        ]
        return self._cell if ranks == list(range(self._cell.num_workers)) else None


class MonarchCellHandle(BaseCellHandle):
    def __init__(self, *, name: str, actor_mesh: Any, num_workers: int) -> None:
        # the name of the proc mesh, which no other launch of the run shares
        self.name = name
        self.actor_mesh = actor_mesh
        self.num_workers = num_workers

    async def call_each(self, method: str, /, kwargs_per_worker: list[dict[str, Any]]) -> list[Any]:
        assert (
            len(kwargs_per_worker) == self.num_workers
        ), f"{len(kwargs_per_worker)} sets of arguments for the {self.num_workers} workers of a cell"
        # Replies are taken as they come, so the first failing worker fails the call at once, as a gather would,
        # rather than once every other worker has answered, which a worker blocked on the failed one never does.
        results: list[Any] = [None] * self.num_workers
        for reply in self.actor_mesh.invoke_each.stream(method, kwargs_per_worker):
            rank, value = await call_endpoint(reply, describe=method)
            results[rank] = value
        return results


async def probe_cell_is_dead(cell_actor_mesh: Any) -> bool:
    """Ask the whole cell, which lives and dies as one: a mesh that lost a process fails a call at once, while a call
    to one worker of it can hang once the mesh is stopped."""
    try:
        await asyncio.wait_for(cell_actor_mesh.ping.call(), timeout=_WAIT_DEAD_PROBE_INTERVAL_SECONDS)
    except (TimeoutError, asyncio.TimeoutError):
        return False
    except (SupervisionError, ActorError):
        return True
    return False


async def call_endpoint(future: Any, *, describe: str) -> Any:
    """Answer what the call answered, raising the actor's own exception the way a Ray actor call does."""
    try:
        return await future
    except (ActorError, SupervisionError) as e:
        raise _caller_error(e, describe=describe) from e


def get_endpoint(future: Any, *, describe: str) -> Any:
    """``call_endpoint`` for a caller that blocks on the answer."""
    try:
        return future.get()
    except (ActorError, SupervisionError) as e:
        raise _caller_error(e, describe=describe) from e


def _caller_error(e: ActorError | SupervisionError, *, describe: str) -> Exception:
    if isinstance(e, SupervisionError):
        return WorkerUnreachableError(f"The actor died or is unreachable when calling {describe!r}: {e!r}")
    if isinstance(e.exception, Exception):
        return e.exception
    # A CancelledError (or other BaseException) raised inside the actor must not read as the caller's own
    return RuntimeError(f"The actor raised {e.exception!r} when calling {describe!r}")
