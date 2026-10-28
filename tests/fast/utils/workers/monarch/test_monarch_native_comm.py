"""--worker-comm-backend monarch on real local procs: workers called through Monarch endpoints, a cell in one call."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("monarch")

from tests.ci.ci_register import register_cpu_ci
from tests.fast.utils.workers.conformance import (
    POOL_ID,
    READY_TIMEOUT_SECONDS,
    SHARED_CHECK_IDS,
    SHARED_CHECKS,
    ConformanceServeSpec,
    ConformanceWorkerConfig,
    HandleCheck,
)
from tests.fast.utils.workers.conftest import worker_manager_args

from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.workers.cell_handle import BaseCellHandle
from miles.utils.workers.monarch.handles import MonarchCellHandle, MonarchWorkerHandle
from miles.utils.workers.monarch.manager import MonarchWorkerManager, create_monarch_backend_capability
from miles.utils.workers.monarch.runtime import GpuLayoutRequest
from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_spec import SchedulingSpec

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["rpc-comm"])

CELL_ID = compute_cell_id(pool_id=POOL_ID, cell_index=0)
NUM_WORKERS = 2

_NO_GPUS = GpuLayoutRequest(num_gpus=0, rollout_offset=0, use_critic=False, num_gpus_per_node=0)


class _TwoWorkerSpec(ConformanceServeSpec):
    def scheduling(self, scaling: ScalingConfig) -> SchedulingSpec:
        return SchedulingSpec(num_cells=1, num_workers_per_cell=NUM_WORKERS, num_gpus_per_worker=0)


@dataclass
class _LaunchedCell:
    manager: Any
    worker_handles: list[BaseWorkerHandle]
    cell_handle: BaseCellHandle


@pytest.fixture(scope="module")
def launched_cell() -> Iterator[_LaunchedCell]:
    manager = MonarchWorkerManager.launch(
        worker_manager_args(env_report_interval_seconds=0.0),
        [_TwoWorkerSpec.create(ConformanceWorkerConfig(rpc_port=0))],
        comm_backend=WorkerCommBackend.MONARCH,
        gpu_layout=_NO_GPUS,
    )
    provider = create_monarch_backend_capability(manager).dynamic_worker_provider(pool_ids=[POOL_ID])
    (infos,) = provider.get_worker_infos(cell_ids=[CELL_ID])
    handles = provider.get_handles_of_worker_infos(infos)
    worker_handles = [handles[info.name] for info in infos]
    cell_handle = provider.get_cell_handle(cell_id=CELL_ID, infos=infos, worker_handles=worker_handles)
    yield _LaunchedCell(manager=manager, worker_handles=worker_handles, cell_handle=cell_handle)
    manager.shutdown.call_one().get()


@pytest.fixture
async def worker_handle(launched_cell: _LaunchedCell) -> AsyncIterator[BaseWorkerHandle]:
    handle = launched_cell.worker_handles[0]
    await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)
    yield handle


class TestAWorkerCalledThroughMonarch:
    def test_the_driver_is_handed_monarch_handles(self, launched_cell: _LaunchedCell):
        """No http server is involved: the handle sends Monarch endpoint calls."""
        assert all(isinstance(handle, MonarchWorkerHandle) for handle in launched_cell.worker_handles)

    @pytest.mark.parametrize("check", SHARED_CHECKS, ids=SHARED_CHECK_IDS)
    async def test_the_handle_contract_holds(self, worker_handle: BaseWorkerHandle, check: HandleCheck):
        """The contract every transport keeps, now over Monarch messages."""
        await check(worker_handle)

    async def test_a_remote_failure_raises_the_worker_s_own_exception(self, worker_handle: BaseWorkerHandle):
        """Like a Ray actor call, a caller catches what the worker raised, not a transport wrapper."""
        with pytest.raises(RuntimeError, match="deliberate"):
            await worker_handle.boom(message="deliberate")


class TestACellCalledInOneMeshCall:
    def test_the_cell_handle_is_a_mesh_call(self, launched_cell: _LaunchedCell):
        assert isinstance(launched_cell.cell_handle, MonarchCellHandle)

    async def test_every_worker_answers_in_worker_order(self, launched_cell: _LaunchedCell):
        tags = await launched_cell.cell_handle.call_each("report_tag", [{}] * NUM_WORKERS)

        assert tags == [f"cell-0-worker-{index}" for index in range(NUM_WORKERS)]

    async def test_each_worker_takes_its_own_arguments(self, launched_cell: _LaunchedCell):
        """One message carries every rank's arguments; each rank picks its own, as a dp rank picks its shard."""
        sums = await launched_cell.cell_handle.call_each("add", [dict(a=index, b=10) for index in range(NUM_WORKERS)])

        assert sums == [10 + index for index in range(NUM_WORKERS)]

    async def test_a_failing_worker_fails_the_whole_call(self, launched_cell: _LaunchedCell):
        with pytest.raises(RuntimeError, match="deliberate"):
            await launched_cell.cell_handle.call_each("boom", [dict(message="deliberate")] * NUM_WORKERS)
