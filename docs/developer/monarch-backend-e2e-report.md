---
title: "Monarch Backend: End-to-End Report"
description: "Ten-step GRPO runs that verify each phase of the Monarch cluster backend against Ray, with timings and caveats."
---

The runs that verified [the Monarch cluster backend](/developer/monarch-backend-design), phase by phase,
against Ray on the same recipe and hardware. The scripts are in `tests/manual/monarch_e2e/`; its README
says how to reproduce them.

## Setup

- **Recipe.** GRPO on Qwen3-0.6B with the FSDP backend: 10 rollouts of 8 prompts × 4 samples, responses up
  to 512 tokens, global batch 32, on a synthetic arithmetic prompt set (`make_dataset.py`).
- **Hardware.** One node, two GB300 gpus (aarch64), SLURM account `sysml`, partition `g3`. The nodes are
  shared with other tenants' jobs.
- **Layouts.** *Colocated*: trainer and engines on both gpus, offloading in turn. *Disaggregated*: one
  trainer gpu, one engine gpu.
- **Commits.** Ray runs: `74524cb4`, the commit before the backend. Monarch runs: the branch as it stood
  when they ran — phase 1 at `e92fc6a3`, phases 2–4 at `a767a894`. A later rebase folded test-only fixups
  into phase 0, so those hashes became `272c45b8` and `6d1a245f`; the code the runs exercised is unchanged.

## Results

Every run finished all 10 train steps and exited 0. Times are averages over steps 2–9 (step 0 and 1 carry
warm-up); throughput is SGLang's reported decode throughput.

| Run | Backend flags | Step | Rollout | Train | Weight update | Decode tok/s | Job |
|---|---|---|---|---|---|---|---|
| Ray, colocated | — | 18.69 s | 15.74 s | 0.93 s | 0.58 s | 392 | 2187362 |
| Monarch phase 1, colocated | `--cluster-backend monarch` (rpc comm, Mooncake) | 18.80 s | 15.76 s | 1.12 s | 0.57 s | 401 | 2187363 |
| Monarch phases 1+3, colocated | `--cluster-backend monarch` (rpc comm, `monarch_rdma` store) | 17.58 s | 14.64 s | 1.01 s | 0.58 s | 457 | 2187545 |
| Monarch phases 1–3, colocated | `… --worker-comm-backend monarch` | 26.14 s | 22.09 s | 1.33 s | 0.90 s | 317 | 2187364 |
| ↳ rerun, rpc comm | as phases 1+3 | 17.74 s | 14.94 s | 0.97 s | 0.58 s | 436 | 2187693 |
| ↳ rerun, monarch comm | as phases 1–3 | 18.97 s | 16.06 s | 1.10 s | 0.56 s | 422 | 2187692 |
| Ray, disaggregated | — (NCCL broadcast) | 3.46 s | 1.08 s | 1.53 s | 0.42 s | 13432 | 2187365 |
| Monarch phases 1–4, disaggregated | `… --worker-comm-backend monarch --update-weight-transfer-mode monarch_rdma` | 3.50 s | 1.23 s | 1.30 s | 0.55 s | 12208 | 2187226 |

## Findings

**Phase 1 matches Ray.** With Monarch only launching and supervising the processes, every timing is within
noise of the Ray baseline: the worker layer above the runtime seam is shared, and the work is the same.

**The RDMA object store costs nothing measurable.** Rollout data handed over as RDMA buffers (phases 1+3) ran
as fast as Mooncake and Ray's object store; the run needed no Mooncake master.

**Native comm is within ~7% of rpc.** The first colocated native-comm run (job 2187364) showed 40% slower
rollouts, and its engines decoded at 317 against 457 tok/s at the same batch sizes. Rollouts are pure
engine work here: the driver's calls through Monarch are not on that path, and a microbenchmark puts one
Monarch call at ~0.5 ms from any event loop. Rerunning both variants back to back with usage sampling
(jobs 2187692 and 2187693) gave 16.1 s against 14.9 s, and the sampling showed other tenants' processes at
100% cpu on the node. Decode is cpu-bound in this setup (CUDA graphs were off, see below), so most of the
first gap is node noise; what remains is within ~7%.

**RDMA weight transfer works and is slower than NCCL on one node.** Every update reached the engine through
the RDMA receiver and SGLang's `update_weights_from_tensor`. After the first, which takes 22.9 s while the
receiver process initialises CUDA and the RDMA manager, updates take 0.55 s against 0.42 s for NCCL
broadcast. On one node NCCL rides NVLink, while this path stages the bucket through host memory on both
sides. Its case is engines on other hosts, which this report does not measure.

## Caveats

The docker image could not be pulled and the `sglang-miles` fork could not be installed, so every run used
a pip-built venv with stock `sglang==0.5.22`. The resulting workarounds applied to Ray and Monarch alike:

- `--sglang-attention-backend triton` and `--sglang-sampling-backend pytorch`, because the default GB300
  kernels download cubins or need a toolchain the venv lacked;
- `--sglang-disable-cuda-graph` on colocated runs, because SGLang's pauseable CUDA graphs need
  `torch_memory_saver`'s preload hook mode;
- `sglang_compat.patch`, a local edit of `miles/utils/weight_version.py` that tolerates a server arg only
  `sglang-miles` defines; it was applied to the test checkouts and never committed.

Single node only: attaching to worker loops on other hosts is implemented but was not run, so neither
multi-node placement nor cross-host RDMA is covered here.
