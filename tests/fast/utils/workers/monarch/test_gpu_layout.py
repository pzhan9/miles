import pytest

pytest.importorskip("monarch")

from miles.utils.workers.monarch.runtime import GpuLayoutRequest, GpuSlot, compute_monarch_gpu_layout


def _request(*, num_gpus: int, rollout_offset: int = 0, use_critic: bool = False) -> GpuLayoutRequest:
    return GpuLayoutRequest(
        num_gpus=num_gpus, rollout_offset=rollout_offset, use_critic=use_critic, num_gpus_per_node=4
    )


class TestComputeMonarchGpuLayout:
    def test_slots_pack_hosts_in_mesh_order(self):
        """Like a PACK placement group: the first host fills up before the second gets a slot."""
        layout = compute_monarch_gpu_layout(
            request=_request(num_gpus=6), host_ips=["10.0.0.2", "10.0.0.1"], num_gpus_per_host=4
        )

        assert layout.slots_of_layout["actor"] == [
            GpuSlot(host_rank=0, host_ip="10.0.0.2", gpu_id=0),
            GpuSlot(host_rank=0, host_ip="10.0.0.2", gpu_id=1),
            GpuSlot(host_rank=0, host_ip="10.0.0.2", gpu_id=2),
            GpuSlot(host_rank=0, host_ip="10.0.0.2", gpu_id=3),
            GpuSlot(host_rank=1, host_ip="10.0.0.1", gpu_id=0),
            GpuSlot(host_rank=1, host_ip="10.0.0.1", gpu_id=1),
        ]

    def test_rollout_slots_start_at_the_rollout_offset(self):
        """A disaggregated run gives the engines the slots after the trainer's, as the rollout placement group does."""
        layout = compute_monarch_gpu_layout(
            request=_request(num_gpus=8, rollout_offset=4), host_ips=["a", "b"], num_gpus_per_host=4
        )

        assert [slot.host_rank for slot in layout.slots_of_layout["rollout"]] == [1, 1, 1, 1]

    def test_colocated_rollout_shares_the_trainer_slots(self):
        layout = compute_monarch_gpu_layout(request=_request(num_gpus=2), host_ips=["a"], num_gpus_per_host=4)

        assert layout.slots_of_layout["rollout"] == layout.slots_of_layout["actor"]

    def test_a_critic_shares_the_actor_slots(self):
        layout = compute_monarch_gpu_layout(
            request=_request(num_gpus=2, use_critic=True), host_ips=["a"], num_gpus_per_host=4
        )

        assert layout.slots_of_layout["critic"] == layout.slots_of_layout["actor"]

    def test_more_gpus_than_the_hosts_offer_is_refused(self):
        with pytest.raises(AssertionError, match="offer 4 each"):
            compute_monarch_gpu_layout(request=_request(num_gpus=5), host_ips=["a"], num_gpus_per_host=4)
