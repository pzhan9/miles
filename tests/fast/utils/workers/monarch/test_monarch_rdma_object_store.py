"""--object-store-backend monarch_rdma on real local procs: a producer process pins, a consumer reads it over rdma."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("monarch")

import torch
from monarch.actor import Actor, endpoint, this_host
from pydantic import TypeAdapter

from tests.ci.ci_register import register_cpu_ci

from miles.utils import object_store
from miles.utils.object_store import MonarchRdmaObjectStore, StoreObjectRef

register_cpu_ci(est_time=40, suite="stage-b-cpu", labels=["rpc-comm"])

_REF_ADAPTER = TypeAdapter(StoreObjectRef)


class _Producer(Actor):
    def __init__(self) -> None:
        object_store.init_instance(SimpleNamespace(object_store_backend="monarch_rdma"))

    @endpoint
    def put(self, value: Any) -> str:
        """Answers the reference as the rpc wire would carry it: json."""
        return _REF_ADAPTER.dump_json(object_store.get_instance().put(value)).decode()

    @endpoint
    def num_pinned(self) -> int:
        from miles.utils.workers.monarch.object_store import _PINNED

        return len(_PINNED)


@pytest.fixture(scope="module")
def producer() -> Iterator[Any]:
    procs = this_host().spawn_procs(per_host={"procs": 1}, name="rdma-producer")
    yield procs.spawn("rdma-producer", _Producer)
    procs.stop().get()


@pytest.fixture(scope="module")
def consumer_store() -> MonarchRdmaObjectStore:
    return MonarchRdmaObjectStore()


def _value() -> dict[str, Any]:
    return {"tokens": [[1, 2, 3], [4, 5]], "rewards": torch.tensor([0.5, -1.0]), "meta": {"rollout_id": 7}}


class TestMonarchRdmaObjectStore:
    def test_a_value_put_in_one_process_is_read_in_another(
        self, producer: Any, consumer_store: MonarchRdmaObjectStore
    ):
        ref = _REF_ADAPTER.validate_json(producer.put.call_one(_value()).get())

        with consumer_store.get(ref) as value:
            assert value["tokens"] == [[1, 2, 3], [4, 5]]
            assert torch.equal(value["rewards"], torch.tensor([0.5, -1.0]))
            assert value["meta"] == {"rollout_id": 7}

    def test_the_reference_survives_the_json_wire(self, producer: Any):
        """Under rpc comm a ref crosses http as json, so it must decode back to a readable reference."""
        wire = producer.put.call_one(_value()).get()

        assert _REF_ADAPTER.dump_json(_REF_ADAPTER.validate_json(wire)).decode() == wire

    def test_remove_releases_the_pinned_bytes_in_the_producer(
        self, producer: Any, consumer_store: MonarchRdmaObjectStore
    ):
        ref = _REF_ADAPTER.validate_json(producer.put.call_one(_value()).get())
        before = producer.num_pinned.call_one().get()

        consumer_store.remove(ref)

        assert producer.num_pinned.call_one().get() == before - 1
