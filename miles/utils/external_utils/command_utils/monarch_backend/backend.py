import os
import shlex

from miles.utils.external_utils.command_utils.base_backend import (
    BaseCommandBackend,
    ExecuteTrainConfig,
    ExecuteTrainRequest,
    LaunchGuard,
)
from miles.utils.external_utils.command_utils.common import (
    _pythonpath_with_sources,
    get_mooncake_master_port,
    run_shell_command,
    train_env_vars,
)
from miles.utils.external_utils.command_utils.ray_backend.command import start_mooncake_master
from miles.utils.external_utils.model_args_utils import shell_safe_model_args
from miles.utils.workers.monarch.hosts import MONARCH_WORKERS_ENV_VAR


class MonarchCommandBackend(BaseCommandBackend):
    """Runs the driver directly as the Monarch client: there is no cluster to start and no job to submit.

    A single-node run spawns every worker on the driver's host. A multi-node run attaches to the Monarch worker
    loops named by ``MILES_MONARCH_WORKERS``, which have to be serving on every host before the launch.
    """

    def _execute_train_inner(
        self, *, request: ExecuteTrainRequest, config: ExecuteTrainConfig, guard: LaunchGuard | None
    ) -> None:
        assert not request.extra_manifests, (
            "extra_manifests are objects a helm release installs beside the run, and a monarch launch installs no "
            "release; launch onto kubernetes, or start what they describe yourself"
        )
        assert config.num_nodes == 1 or os.environ.get(MONARCH_WORKERS_ENV_VAR), (
            f"a {config.num_nodes}-node monarch run attaches to the worker loops of its hosts; start "
            f"`python -m miles.utils.workers.monarch.worker_loop` on each and set {MONARCH_WORKERS_ENV_VAR}"
        )
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")

        self._clean_up_previous_run()
        if not config.external_mooncake:
            start_mooncake_master(
                rpc_port=get_mooncake_master_port(shlex.split(request.train_args)), run_command=self.exec_command_cpu
            )
        for cmd in request.prepare_cmd.values():
            self.exec_command_multi_node(cmd)
        if (f := request.before_ray_job_submit) is not None:
            f()

        env_vars = train_env_vars(
            request, {"no_proxy": f"127.0.0.1,{master_addr}", "MASTER_ADDR": master_addr}, config=config
        )
        env_vars["PYTHONPATH"] = _pythonpath_with_sources(request.megatron_path, env_vars.get("PYTHONPATH"))
        model_args = shell_safe_model_args(request.megatron_model_type)
        exports = " ".join(f"{name}={shlex.quote(value)}" for name, value in env_vars.items())
        self.exec_command_cpu(
            f"env {exports} python3 {shlex.quote(request.train_script)} {model_args} {request.train_args}"
        )

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
        assert not os.environ.get(MONARCH_WORKERS_ENV_VAR), (
            "running a command on every attached monarch host is not supported yet; run it on each host before "
            "starting its worker loop"
        )
        return [run_shell_command(cmd, capture_output=capture_output)]

    def _clean_up_previous_run(self) -> None:
        self.exec_command_cpu("pkill -9 sglang; sleep 3; pkill -9 miles; pkill -9 mooncake_master; true")
