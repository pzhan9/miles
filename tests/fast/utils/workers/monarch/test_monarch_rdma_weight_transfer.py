"""--update-weight-transfer-mode monarch_rdma on real local procs: a receiver next to each engine pulls the bucket."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("monarch")
pytest.importorskip("sglang")

import torch
from monarch.actor import this_host

from tests.ci.ci_register import register_cpu_ci

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.utils.workers.monarch.runtime import MonarchWorkerActor, _WorkerTarget
from miles.utils.workers.monarch.weight_transfer import push_bucket

register_cpu_ci(est_time=40, suite="stage-b-cpu", labels=["rpc-comm"])

NUM_ENGINES = 2


class _Idle:
    """What an engine cell's process hosts besides the receiver: here, nothing."""


@dataclass(frozen=True)
class _FakeEngine(SGLangApiClient):
    """Answers like an engine on cpu and imports the bucket the way SGLang's flattened_bucket loader does."""

    async def get_server_info(self) -> dict[str, Any]:
        return {"device": "cpu", "base_gpu_id": 0}

    async def update_weights_from_tensor(self, serialized_named_tensors, load_format=None, selector="all", **_):
        from sglang.srt.utils import MultiprocessingSerializer
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

        assert load_format == "flattened_bucket"
        [payload] = serialized_named_tensors
        data = MultiprocessingSerializer.deserialize(payload)
        bucket = FlattenedTensorBucket(flattened_tensor=data["flattened_tensor"], metadata=data["metadata"])
        return {
            "success": True,
            "engine": self.server_url,
            "tensors": {name: tensor.tolist() for name, tensor in bucket.reconstruct_tensors()},
        }


@pytest.fixture(scope="module")
def engine_processes() -> Iterator[list[Any]]:
    procs = this_host().spawn_procs(per_host={"procs": NUM_ENGINES}, name="engine-cells")
    actors = procs.spawn(
        "engine-cells",
        MonarchWorkerActor,
        [_WorkerTarget(actor_class=_Idle, ctor_kwargs={}, served_natively=False)] * NUM_ENGINES,
        None,
    )
    yield [actors.slice(procs=index) for index in range(NUM_ENGINES)]
    procs.stop().get()


def test_every_engine_imports_the_bucket_the_trainer_staged(engine_processes: list[Any]):
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

    named = [("w", torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)), ("b", torch.tensor([7.0, 8.0]))]
    bucket = FlattenedTensorBucket(named_tensors=named)
    engines = [_FakeEngine(server_url=f"http://engine-{index}") for index in range(NUM_ENGINES)]

    results = push_bucket(
        flattened=bucket.get_flattened_tensor(),
        metadata=bucket.get_metadata(),
        engine_processes=engine_processes,
        engines=engines,
        selector="all",
    )

    assert [result["engine"] for result in results] == [engine.server_url for engine in engines]
    assert all(result["tensors"] == {"w": [[0, 1, 2], [3, 4, 5]], "b": [7.0, 8.0]} for result in results)
