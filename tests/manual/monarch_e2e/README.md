# Monarch backend end-to-end runs

The scripts behind [the Monarch backend e2e report](../../../docs/developer/monarch-backend-e2e-report.md): one
10-step GRPO run of Qwen3-0.6B (FSDP) on two gpus per configuration, on a SLURM cluster whose compute nodes
have no network. Nothing here is collected by CI.

| File | What it does |
|---|---|
| `build_env.sh` | Builds the aarch64 / CUDA 13 venv from PyPI and writes `$E2E_ROOT/env.sh` |
| `make_dataset.py` | Writes the arithmetic prompt set the runs train on |
| `e2e.sbatch` | One run; the backend flags come in as extra train args |
| `run_matrix.sh` | Submits every run of the report |
| `sglang_compat.patch` | Local-only shim for stock sglang, see below |

## Reproduce

```bash
export E2E_ROOT=~/miles_e2e
PYTHON=<python3.12 with headers> bash tests/manual/monarch_e2e/build_env.sh
python tests/manual/monarch_e2e/make_dataset.py $E2E_ROOT/data/arith.jsonl

git worktree add $E2E_ROOT/wt-base <commit before the backend>
git worktree add $E2E_ROOT/wt-monarch monarch
for wt in $E2E_ROOT/wt-base $E2E_ROOT/wt-monarch; do (cd $wt && patch -p1 < <repo>/tests/manual/monarch_e2e/sglang_compat.patch); done

BASE_SRC=$E2E_ROOT/wt-base MONARCH_SRC=$E2E_ROOT/wt-monarch bash tests/manual/monarch_e2e/run_matrix.sh
```

Every run ends its log with `EXIT CODE <n>`; a train step logs `step <n>: {...}` and its timings as
`perf/step_time`, `perf/rollout_time` and `perf/update_weights_time`.

## Environment workarounds

The docker image could not be pulled and the `sglang-miles` fork could not be installed, so the venv carries
stock `sglang==0.5.22`. Each workaround applies to the Ray and the Monarch runs alike:

- `--sglang-attention-backend triton` and `--sglang-sampling-backend pytorch`: the default kernels on GB300
  download cubins or need a toolchain the venv does not have.
- `--sglang-disable-cuda-graph` for colocated runs: SGLang's pauseable CUDA graphs need
  `torch_memory_saver`'s preload hook mode, which the PyPI build lacks.
- `sglang_compat.patch`: `miles/utils/weight_version.py` reads `enable_prefill_weight_versions`, a server arg
  only `sglang-miles` has. Apply it to the test checkouts only; never commit it.
