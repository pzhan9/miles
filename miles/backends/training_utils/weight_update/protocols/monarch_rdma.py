from argparse import Namespace
from collections.abc import Sequence

import torch

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.protocols.utils.data_replica import get_data_replica_rank_and_size
from miles.backends.training_utils.weight_update.rollout_cell_updater import create_rollout_cell_updaters
from miles.backends.training_utils.weight_update.session import check_weight_sync_results
from miles.utils.workers.monarch.weight_transfer import engine_processes_of_cells, push_bucket


class UpdateWeightFromMonarchRdma(WeightTransferProtocol):
    """Non-colocated transfer under --cluster-backend monarch: one sender per replica set stages each flattened
    bucket for RDMA, and a receiver next to every engine pulls it and imports it through CUDA IPC."""

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self._engine_processes: list = []

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        engine_cell_ids: Sequence[str],
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        self.rollout_engines = rollout_engines
        self.cell_updaters_of_cell_id = create_rollout_cell_updaters(self.args, self.rollout_engines, engine_cell_ids)
        self._selector = selector
        assert engine_gpu_counts is not None and all(count == 1 for count in engine_gpu_counts), (
            f"--update-weight-transfer-mode monarch_rdma imports a bucket on one engine gpu, but the engines span "
            f"{engine_gpu_counts} gpus"
        )

        replica_rank, _ = get_data_replica_rank_and_size(parallel_state, placement)
        self.is_sender = replica_rank == 0
        if self.is_sender:
            self._engine_processes = engine_processes_of_cells(list(engine_cell_ids))

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        # sglang is imported only by runs that update engine weights
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

        flattened_bucket = FlattenedTensorBucket(
            named_tensors=[(name, tensor.contiguous()) for name, tensor in bucket]
        )
        results = push_bucket(
            flattened=flattened_bucket.get_flattened_tensor(),
            metadata=flattened_bucket.get_metadata(),
            engine_processes=self._engine_processes,
            engines=list(self.rollout_engines),
            selector=self._selector,
        )
        check_weight_sync_results(results, is_lora=False)
        bucket.clear()
