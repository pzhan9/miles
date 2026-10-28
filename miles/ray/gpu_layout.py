"""The gpu slots a run lays its gpu workers out on, independent of the cluster backend that provides them."""

import socket

from miles.ray.specs.train import ACTOR_ROLE, compute_trainer_configs
from miles.utils.workers.types import DeployComponent


def node_gpu_sort_key(x) -> tuple[list[int], int]:
    """Order (index, node identifier, gpu id) triples by node address, then gpu id, so slots pack node by node."""
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)


def compute_gpu_layout(args) -> tuple[int, int]:
    """(gpus the run places its gpu workers on, index of the first rollout slot among them)."""
    selector = DeployComponent(args.deploy_component)
    trainer_num_gpus = _compute_trainer_num_gpus(args) if selector.selects(DeployComponent.TRAINER) else 0
    if args.debug_train_only:
        eval_num_gpus = args.eval_num_gpus if selector.selects(DeployComponent.INFERENCE) else 0
        return trainer_num_gpus + eval_num_gpus, trainer_num_gpus

    rollout_num_gpus = (
        (args.rollout_num_gpus or 0) + args.eval_num_gpus if selector.selects(DeployComponent.INFERENCE) else 0
    )
    if args.rollout_external:
        return (0, 0) if args.debug_rollout_only else (trainer_num_gpus, trainer_num_gpus)
    if args.debug_rollout_only:
        return rollout_num_gpus, 0
    if args.colocate:
        return max(trainer_num_gpus, rollout_num_gpus), 0
    return trainer_num_gpus + rollout_num_gpus, trainer_num_gpus


def _compute_trainer_num_gpus(args) -> int:
    num_policies = len([config for config in compute_trainer_configs(args) if config.role == ACTOR_ROLE])
    return args.actor_num_nodes * args.actor_num_gpus_per_node * num_policies
