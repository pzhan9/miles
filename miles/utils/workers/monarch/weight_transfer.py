"""Weight buckets pulled by the engines over Monarch RDMA: ``--update-weight-transfer-mode monarch_rdma``.

The trainer's sending rank stages each flattened bucket in pinned host memory and hands every engine an
``RDMABuffer`` over it. A receiver runs inside the process Monarch launched for the engine's cell, on the engine's
host: it pulls the bucket with one one-sided read, lands it on the engine's gpu and gives SGLang a CUDA IPC handle
to it, exactly as a colocated trainer would. No NCCL group spans the trainer and the engines, so neither side
has to join a collective, and the engines read from the trainer as fast as the fabric lets them.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
from monarch.rdma import RDMABuffer

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.utils.misc import to_local_gpu_id
from miles.utils.workers.monarch.handles import get_endpoint_result
from miles.utils.workers.monarch.runtime import installed_manager_handle

_READ_TIMEOUT_SECONDS = 600


def engine_processes_of_cells(cell_ids: list[str]) -> list[Any]:
    """The Monarch actor hosting each engine cell's one process, where its receiver runs."""
    manager = installed_manager_handle()
    processes = []
    for cell_id in cell_ids:
        infos = manager.get_worker_infos.call_one(cell_id).get()
        assert len(infos) == 1, (
            f"monarch_rdma weight transfer lands a bucket on the gpu of a one-process engine, but cell {cell_id} has "
            f"{len(infos)} processes"
        )
        [info] = infos
        processes.append(manager.get_actor_handle.call_one(info.name, expected_generation=info.generation).get())
    return processes


def push_bucket(
    *,
    flattened: torch.Tensor,
    metadata: Any,
    engine_processes: list[Any],
    engines: list[SGLangApiClient],
    selector: str,
) -> list[Any]:
    """Stage one flattened bucket, let every engine pull it, and answer each engine's update result."""
    staging = torch.empty(flattened.numel() * flattened.element_size(), dtype=torch.uint8, pin_memory=True)
    staging.copy_(flattened.contiguous().view(-1).view(torch.uint8))
    buffer = RDMABuffer(staging)
    try:
        receive = partial(
            receive_bucket,
            buffer=buffer,
            num_bytes=staging.numel(),
            dtype=flattened.dtype,
            metadata=metadata,
            selector=selector,
        )
        futures = [
            process.run_in_process.call_one(partial(receive, engine=engine))
            for process, engine in zip(engine_processes, engines, strict=True)
        ]
        return [get_endpoint_result(future, describe="receive_bucket") for future in futures]
    finally:
        buffer.drop().get()


async def receive_bucket(
    *,
    buffer: RDMABuffer,
    num_bytes: int,
    dtype: torch.dtype,
    metadata: Any,
    selector: str,
    engine: SGLangApiClient,
) -> Any:
    # sglang is imported only by the processes that receive weights
    from sglang.srt.utils import MultiprocessingSerializer

    staging = torch.empty(num_bytes, dtype=torch.uint8)
    await buffer.read_into(staging, timeout=_READ_TIMEOUT_SECONDS)
    flattened = staging.to(await _device_of(engine)).view(dtype)
    serialized = MultiprocessingSerializer.serialize(
        {"flattened_tensor": flattened, "metadata": metadata}, output_str=True
    )
    # The engine maps the bucket through its IPC handle, so it stays alive here until the engine answers.
    result = await engine.update_weights_from_tensor(
        serialized_named_tensors=[serialized], load_format="flattened_bucket", selector=selector
    )
    on_cuda = flattened.is_cuda
    del flattened
    if on_cuda:
        torch.cuda.ipc_collect()
    return result


_DEVICE_OF_ENGINE: dict[str, torch.device] = {}


async def _device_of(engine: SGLangApiClient) -> torch.device:
    if (device := _DEVICE_OF_ENGINE.get(engine.server_url)) is None:
        info = await engine.get_server_info()
        device_type = info.get("device", "cuda")
        device = (
            torch.device(device_type)
            if device_type == "cpu"
            else torch.device(device_type, to_local_gpu_id(int(info["base_gpu_id"])))
        )
        _DEVICE_OF_ENGINE[engine.server_url] = device
    return device
