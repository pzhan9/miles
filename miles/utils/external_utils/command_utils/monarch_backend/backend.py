import os
import shlex

from miles.utils.external_utils.command_utils.base_backend import (
    CLUSTER_BACKEND_FLAG,
    BaseCommandBackend,
    ExecuteTrainConfig,
    ExecuteTrainRequest,
    LaunchGuard,
)
from miles.utils.external_utils.command_utils.common import (
    ArgvManipulator,
    _pythonpath_with_sources,
    get_mooncake_master_port,
    run_shell_command,
    train_env_vars,
)
from miles.utils.external_utils.command_utils.ray_backend.command import start_mooncake_master
from miles.utils.external_utils.model_args_utils import shell_safe_model_args
from miles.utils.workers.types import ClusterBackend


class MonarchCommandBackend(BaseCommandBackend):
    """Runs the driver directly as the Monarch client: there is no cluster to start and no job to submit.

    The run spawns every worker on the driver's host.
    """

    def _execute_train_inner(
        self, *, request: ExecuteTrainRequest, config: ExecuteTrainConfig, guard: LaunchGuard | None
    ) -> None:
        assert not request.extra_manifests, (
            "extra_manifests are objects a helm release installs beside the run, and a monarch launch installs no "
            "release; launch onto kubernetes, or start what they describe yourself"
        )
        assert config.num_nodes == 1, f"a monarch run spawns its workers on one host, not {config.num_nodes} nodes"
        master_addr = os.environ.get("MASTER_ADDR") or "127.0.0.1"
        mooncake_master_port = get_mooncake_master_port(shlex.split(request.train_args))

        self._clean_up_previous_run()
        if not config.external_mooncake:
            start_mooncake_master(rpc_port=mooncake_master_port, run_command=self.exec_command_cpu)
        for cmd in request.prepare_cmd.values():
            self.exec_command_multi_node(cmd)
        if (f := request.before_ray_job_submit) is not None:
            f()

        backend_env_vars = {"no_proxy": f"127.0.0.1,{master_addr}", "MASTER_ADDR": master_addr}
        env_vars = train_env_vars(request, backend_env_vars, config=config)
        env_vars["PYTHONPATH"] = _pythonpath_with_sources(request.megatron_path, env_vars.get("PYTHONPATH"))
        model_args = shell_safe_model_args(request.megatron_model_type)
        train_args = request.train_args
        # train.py defaults to --cluster-backend ray, whose cluster this launch never starts
        if not ArgvManipulator.is_defined(shlex.split(train_args), CLUSTER_BACKEND_FLAG):
            train_args = f"{train_args} {CLUSTER_BACKEND_FLAG} {ClusterBackend.MONARCH.value}"
        exports = " ".join(f"{name}={shlex.quote(value)}" for name, value in env_vars.items())
        self.exec_command_cpu(f"env {exports} python3 {shlex.quote(request.train_script)} {model_args} {train_args}")

    def _exec_command_gpu_inner(
        self, cmd: str, capture_output: bool = False, num_gpus_per_node: int | None = None
    ) -> str | None:
        return run_shell_command(cmd, capture_output=capture_output)

    def _exec_command_multi_node_inner(
        self,
        cmd: str,
        capture_output: bool = False,
        num_nodes: int | None = None,
        num_gpus_per_node: int | None = None,
    ) -> list[str | None]:
        return [run_shell_command(cmd, capture_output=capture_output)]

    def _clean_up_previous_run(self) -> None:
        self.exec_command_cpu("pkill -9 sglang; sleep 3; pkill -9 miles; pkill -9 mooncake_master; true")
