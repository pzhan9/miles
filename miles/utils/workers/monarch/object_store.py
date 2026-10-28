"""``--object-store-backend monarch_rdma``: rollout data handed over as Monarch RDMA buffers.

``put`` pins the serialized value in the producing process and answers a reference to it; ``get`` reads it with
one one-sided RDMA read straight out of the producer's memory, so the bytes never pass through the driver or an
external store. The producer keeps the bytes until ``remove`` asks the holder actor of its process to release them.
"""

from __future__ import annotations

import logging
import pickle
import threading
import uuid
from dataclasses import dataclass
from typing import Any

import monarch
import torch
from monarch.actor import Actor, endpoint, this_proc
from monarch.rdma import RDMABuffer

logger = logging.getLogger(__name__)

_HOLDER_NAME = "miles-object-holder"
_READ_TIMEOUT_SECONDS = 600


@dataclass(frozen=True, kw_only=True)
class MonarchRdmaRef:
    key: str
    num_bytes: int
    buffer: RDMABuffer
    holder: Any


# What this process pinned for others to read, by key; the holder actor of this process releases entries.
_PINNED: dict[str, tuple[torch.Tensor, RDMABuffer]] = {}
_PINNED_LOCK = threading.Lock()


class _ObjectHolder(Actor):
    @endpoint
    def release(self, key: str) -> None:
        with _PINNED_LOCK:
            pinned = _PINNED.pop(key, None)
        if pinned is not None:
            pinned[1].drop().get()


class MonarchRdmaStore:
    def __init__(self) -> None:
        # ibverbs where the hosts have it; elsewhere (a cpu test host) the same calls go over tcp
        monarch.configure(rdma_allow_tcp_fallback=True)
        self._holder = this_proc().spawn(_HOLDER_NAME, _ObjectHolder)

    def put(self, value: Any) -> MonarchRdmaRef:
        payload = torch.frombuffer(bytearray(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)), dtype=torch.uint8)
        key = uuid.uuid4().hex
        buffer = RDMABuffer(payload)
        with _PINNED_LOCK:
            _PINNED[key] = (payload, buffer)
        return MonarchRdmaRef(key=key, num_bytes=payload.numel(), buffer=buffer, holder=self._holder)

    def get(self, ref: MonarchRdmaRef) -> Any:
        payload = torch.empty(ref.num_bytes, dtype=torch.uint8)
        ref.buffer.read_into(payload, timeout=_READ_TIMEOUT_SECONDS).get()
        return pickle.loads(payload.numpy().tobytes())

    def remove(self, ref: MonarchRdmaRef) -> None:
        try:
            ref.holder.release.call_one(ref.key).get()
        except Exception:
            logger.warning(f"Releasing object {ref.key} failed; its producer is likely gone", exc_info=True)
