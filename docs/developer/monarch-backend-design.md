---
title: "Design: Monarch Cluster Backend"
description: "How Miles runs on Monarch instead of Ray: the worker-manager seam, one proc mesh per cell, mesh calls, and RDMA data planes."
---

**Status:** implemented, experimental. **Scope:** `--cluster-backend monarch`, picked per run as an
alternative to Ray; Ray stays the default. User-facing instructions are in
[Ray, Monarch and Kubernetes Backend](/advanced/cluster-backend).

## 1. Summary

Miles chooses its cluster backend at runtime, and everything above `miles/utils/workers/` talks to
workers through backend-neutral interfaces (`BackendCapability`, `BaseWorkerProvider`,
`BaseCellOperations`, `BaseWorkerHandle`). [Monarch](https://github.com/meta-pytorch/monarch) plugs in
at three levels, each selected by its own flag:

| Phase | What Monarch does | Flag | Default under `monarch` |
|---|---|---|---|
| 0 | Nothing yet: the Ray worker manager is split into a backend-neutral core and a Ray runtime | — | — |
| 1 | Starts, places, supervises and stops every worker; one `ProcMesh` per cell | `--cluster-backend monarch` | — |
| 2 | Carries the driver's calls to workers; a whole cell is one mesh call | `--worker-comm-backend monarch` | `rpc` |
| 3 | Hands rollout data over as RDMA buffers read straight out of the producer | `--object-store-backend monarch_rdma` | `monarch_rdma` |
| 4 | Lets each engine pull weight buckets from the trainer over RDMA | `--update-weight-transfer-mode monarch_rdma` | `broadcast` |

Phase 1 alone is "Monarch instead of `ray start`". Phases 2–4 are where Monarch's own strengths — mesh
calls, supervision and one-sided RDMA — replace HTTP fan-out, Mooncake and NCCL broadcast.

## 2. Goals and non-goals

**Goals**

- One flag picks the backend; specs, recipes and the training loop are unchanged.
- Same placement and failure semantics as the Ray worker manager: the same gpu packing and colocation,
  and a cell that loses a process is torn down and reported as not alive.
- `torchmonarch` is an optional extra (`pip install miles[monarch]`); a Ray run never imports it.
- No second copy of the pool/cell bookkeeping ([SSOT](../../.claude/rules/single-source-of-truth.md)).

**Non-goals**

- Removing Ray. Monarch's tensor engine (Megatron/FSDP keep their own `torch.distributed`).
- The dashboard, Prometheus collector, distributed post and multi-LoRA, which need a Ray cluster and are
  refused under `monarch` as under `kubernetes`. Hot restart, which is Kubernetes-only.

## 3. Background: where Ray was

- `RayWorkerManager` (`miles/utils/workers/ray_worker_manager.py`) did two jobs at once: the
  backend-neutral bookkeeping (pools, cells, generations, ports, liveness, fault hooks) and the Ray
  mechanics (placement groups, `ray.remote` options, `ray.kill`, `__ray_ready__`, concurrency groups, the
  named-actor lookup workers use to reach it).
- A handful of node probes (`get_current_node_ip`, the trainer's gpu id) called into Ray only to learn
  where they run.
- `tests/fast/utils/workers/test_ray_communication_boundary.py` is the ledger of every module allowed to
  import Ray.

## 4. Monarch, as Miles uses it

- `HostMesh.spawn_procs(per_host=..., bootstrap_command=...)` → `ProcMesh`; `ProcMesh.spawn(...)` →
  `ActorMesh`; `@endpoint` methods called with `.call()` (whole mesh, answered as a `ValueMesh`),
  `.call_one()`, `.broadcast()`.
- Monarch assigns no gpus; a per-coordinate `bootstrap_command` sets each process's environment.
- Every mesh is owned by the actor that spawned it; failures arrive at the owner's `__supervise__`.
- There is no named-actor registry and no object store; `monarch.rdma.RDMABuffer` exposes 1-D contiguous
  memory for one-sided `read_into` (ibverbs, with a tcp fallback).

Miles imports only `monarch.actor`, `monarch.rdma` and, for `SupervisionError`, one rust binding; every
import lives in `miles/utils/workers/monarch/`, which `test_monarch_boundary.py` enforces.

## 5. Phase 0: the worker-manager seam

`miles/utils/workers/worker_manager/` holds what both backends share:

| Module | Role |
|---|---|
| `core.py` | `WorkerManager`: pools, cells, generations, port allocation, liveness scan, fault hooks; the command / rpc / native serve paths |
| `runtime.py` | `WorkerRuntime` (`launch_cell`, `kill_worker`, `release_cell`, `physical_gpu_ids`, `worker_capability_factory`, `cell_native_handle`) and `WorkerProcess` (`call`, `call_without_result`, `probe_is_dead`, `native_handle`) |
| `client.py` | `WorkerManagerClient` and the provider / cell operations / capability built on it, written once for every backend |

`RayWorkerManager` is now the core plus `RayWorkerRuntime`; the Ray provider, cell operations and
capability are thin `RayWorkerManagerClient` wrappers. Alongside:

- `ClusterBackend` predicates (`provides_ray_cluster`, `launches_workers_from_driver`,
  `auto_resumes_cells`) replace every `== KUBERNETES`, so each site states the reason it cares.
- `get_current_node_ip` no longer calls into Ray; the manager hands every gpu worker its physical gpu ids
  in `MILES_WORKER_GPU_IDS`.
- The gpu layout math moved to `miles/ray/gpu_layout.py`, shared by placement groups and Monarch.

Deviation from the proposal: the `RAY_EXPERIMENTAL_NOSET_*` variables stay in the specs that own them.
Moving them would have changed every Kubernetes chart snapshot for no behaviour gain; under Monarch they
are inert.

## 6. Phase 1: Monarch launches the run

**Manager.** The driver spawns `MonarchWorkerManager`, a Monarch actor in its own process on the driver's
host, which wraps the backend-neutral core and owns every cell's mesh
(`miles/utils/workers/monarch/manager.py`). Workers reach it through a handle installed into each process
they start (`install_manager_handle`), since Monarch has no registry.

**Placement.** `compute_gpu_layout` (the numbers Ray's placement groups get) arrives as a
`GpuLayoutRequest`; `compute_monarch_gpu_layout` packs logical slots onto hosts in mesh order, like a
`PACK` placement group, after probing every host's address and visible gpus. Gpu workers keep their
whole host visible and select the devices the manager placed them on, as under Ray; cpu-only workers see
none (`CUDA_VISIBLE_DEVICES=""`).

**One `ProcMesh` per cell.** `MonarchWorkerRuntime.launch_cell` spawns the cell's processes on its run of
consecutive hosts in one `spawn_procs`, with each rank's environment from `bootstrap_command`, and one
`MonarchWorkerActor` per rank hosting the `CommandActor` / `ServeActor` / served worker the core asks for.
Every rank receives the targets of the whole cell and builds the one at its rank. A cell whose workers
are not an equal number on consecutive hosts is refused; Kubernetes already requires the same shape.

**Failure.** `MonarchWorkerManager.__supervise__` maps a failed mesh to its cell and tears it down through
the same generation-checked path the liveness scan takes (`WorkerManager.report_lost_cell`), then reports
the failure handled so the run survives. The ping-based scan stays as a backstop.

**Hosts and launcher.** The driver's host by default; a multi-node run attaches to worker loops started
with `python -m miles.utils.workers.monarch.worker_loop` and named in `MILES_MONARCH_WORKERS`.
`MonarchCommandBackend` runs the driver directly as the Monarch client — there is no cluster to start and
no job to submit.

## 7. Phase 2: cells as meshes

Under `--worker-comm-backend monarch` the driver calls workers through Monarch endpoints:

- `BaseCellHandle.call_each(method, kwargs_per_worker)` calls the same method on every worker of a cell
  and answers in worker order. `GatherCellHandle` gathers per-worker calls (every other transport);
  `MonarchCellHandle` is **one** `ActorMesh.call`, fanned out by Monarch and answered as a `ValueMesh`.
- `TrainerCell.execute` goes through its cell handle, so `init`, `load_state`, `train` and the rest are one
  message per cell. The arguments are identical on every rank except where a rank takes its share (each dp
  rank still fetches its own shard ref), so every rank picks its entry out of the one message.
- A natively served worker runs each call through the same `RpcCallExecutor` the rpc server uses, so
  concurrency groups, sync and async methods and dedicated-loop workers behave exactly as over rpc.
- `MonarchWorkerHandle` re-raises the worker's own exception, as a Ray call does, and turns a supervision
  error into `WorkerUnreachableError`.

Across cells nothing changes: each cell has its own state machine and failure domain, so the trainer
controller still gathers over cells, one mesh call inside each.

## 8. Phase 3: RDMA object store

`--object-store-backend monarch_rdma` (`miles/utils/workers/monarch/object_store.py`):

- `put` pickles the value into a pinned buffer in the producing process and answers a reference holding
  the `RDMABuffer`, its size, a key and the holder actor of that process.
- `get` is one one-sided read straight out of the producer's memory: the bytes never pass through the
  orchestrator or an external store.
- `remove` asks the producer's holder actor to drop the buffer.

`rollout_data_ref` and every caller are unchanged; the ref's payload is pickled to a string so the rpc
wire carries it as json. Without a Ray cluster, `ray` now resolves to `mooncake` under Kubernetes and to
`monarch_rdma` under Monarch, so a Monarch run needs no Mooncake master; `mooncake` stays selectable.

## 9. Phase 4: RDMA weight transfer

`--update-weight-transfer-mode monarch_rdma` (`protocols/monarch_rdma.py`,
`miles/utils/workers/monarch/weight_transfer.py`), for non-colocated runs:

1. The sending rank of each replica set flattens a bucket (`FlattenedTensorBucket`), stages it in pinned
   host memory and creates one `RDMABuffer` over it.
2. A receiver runs in the process Monarch launched for each engine's cell, on the engine's host
   (`MonarchWorkerActor.run_in_process`). It pulls the bucket with one read, lands it on the engine's
   device and hands SGLang a CUDA IPC handle through `update_weights_from_tensor` (`flattened_bucket`) —
   the path a colocated trainer takes.
3. No NCCL group spans trainer and engines, so neither side waits in a collective for the other.

Limit: one-gpu engines (a multi-gpu engine would need one receiver per tp rank).

## 10. Testing

| What | Where |
|---|---|
| The Ray worker manager, unchanged in behaviour | `tests/fast/utils/workers/test_ray_worker_manager*.py`, `real_ray/` |
| The rpc conformance contract, supervision teardown, stop/start on real local Monarch procs | `tests/fast/utils/workers/monarch/test_monarch_worker_manager.py` |
| The shared handle contract over Monarch endpoints, mesh-call ordering, per-rank arguments, failure propagation | `test_monarch_native_comm.py` |
| Put in one process and read in another, the json wire, release in the producer | `test_monarch_rdma_object_store.py` |
| Two engine-cell processes pull and import a mixed-dtype bucket | `test_monarch_rdma_weight_transfer.py` |
| Gpu layout, backend predicates, object-store resolution, import boundary | `test_gpu_layout.py`, `test_types.py`, `test_object_store_resolution.py`, `test_monarch_boundary.py` |

All Monarch tests run on cpu with local procs (RDMA falls back to tcp where there is no ibverbs).

### End-to-end verification

GRPO on Qwen3-0.6B (FSDP, 10 rollouts of 8 prompts × 4 samples, 512-token responses) on one node with
two GB300 gpus. Every run finished all 10 train steps and exited 0. Times are averages over steps 2–9.

| Run | Flags beyond the recipe | Step | Rollout | Weight update |
|---|---|---|---|---|
| Ray, colocated (baseline) | — | 18.7 s | 15.7 s | 0.58 s |
| Monarch phase 1, colocated | `--cluster-backend monarch --object-store-backend mooncake` | 18.8 s | 15.8 s | 0.57 s |
| Monarch phases 1–3, colocated | `… --worker-comm-backend monarch` (object store `monarch_rdma`) | 26.1 s ¹ | 22.1 s ¹ | 0.90 s |
| Monarch phases 1+3, colocated | `--cluster-backend monarch` (rpc comm, `monarch_rdma`) | — | 14.6 s | — |
| Ray, disaggregated (baseline) | 1 trainer gpu + 1 engine gpu, NCCL broadcast | 3.46 s | — | 0.42 s |
| Monarch phases 1–4, disaggregated | `… --worker-comm-backend monarch --update-weight-transfer-mode monarch_rdma` | 3.50 s | — | 0.55 s ² |

¹ A rerun with usage sampling measured 16.1 s of rollout for native comm against 14.9 s for rpc comm on
the same commit; the gpu nodes are shared with other tenants, and decode is cpu-bound here (CUDA graphs
were off, see below), so most of the first gap is node noise. Native comm stays within ~7% of rpc.

² The first update takes 22.9 s (the receiver process initialises CUDA and the RDMA manager); later ones
take 0.55 s against 0.42 s for NCCL. On one node NCCL rides NVLink while this path stages through host
memory on both sides; it is meant for engines on other hosts, where NCCL would need a group spanning the
trainer and every engine.

The environment was a pip-built aarch64 venv (stock `sglang==0.5.22`, no `sglang-miles`), which needed
three settings unrelated to Monarch, identical for the Ray and Monarch runs: `--sglang-attention-backend
triton`, `--sglang-sampling-backend pytorch` and, for colocated runs, `--sglang-disable-cuda-graph`; and
one local, uncommitted read of a `sglang-miles`-only server arg in `miles/utils/weight_version.py`.

## 11. Open items

- Multi-gpu engines under `monarch_rdma` weight transfer; a receiver per tp rank.
- Running a command on every attached host from the launcher (`exec_command_multi_node`) for multi-node
  runs; today it runs on the driver's host only.
- A supervision-driven respawn of a failed cell inside the manager, instead of waiting for the reconcile
  loop to resume it.
- Monarch is Beta and its internals change; `SupervisionError` is imported from a rust binding until it is
  exported publicly.
