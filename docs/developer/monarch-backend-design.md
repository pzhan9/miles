---
title: "Design: Monarch Cluster Backend"
description: "Proposal for adding Monarch as a cluster backend users can pick instead of Ray, reusing the worker layer Kubernetes already proved out."
---

**Status:** proposal, not implemented. **Scope:** `--cluster-backend monarch`, picked per run as an
alternative to Ray.

## 1. Summary

Miles already chooses its cluster backend at runtime. `--cluster-backend` takes `ray` (the default) or
`kubernetes`. Everything above `miles/utils/workers/` talks to workers through backend-neutral
interfaces: `BackendCapability`, `BaseWorkerProvider`, `BaseCellOperations` and `BaseWorkerHandle`.
On top of that, `--worker-comm-backend rpc` gives every worker an HTTP RPC server that does not depend
on Ray.

The proposal is to add **`ClusterBackend.MONARCH`**. [Monarch](https://github.com/meta-pytorch/monarch)
would replace Ray as the thing that **starts, places, watches and stops worker processes**. The rest of
the run stays as it is: specs, pools, cells, the reconcile loop, the RPC wire and the training loop.

The work comes in three phases. Each phase can ship and be tested on its own.

| Phase | What changes | Comm backend | Object store |
|---|---|---|---|
| 0 | Split `RayWorkerManager` into a backend-neutral core plus a Ray runtime. No behaviour change. | `ray` / `rpc` | unchanged |
| 1 | `MonarchWorkerRuntime` and launcher. A full run on Monarch. | `rpc` (reused) | `mooncake` (reused) |
| 2 (optional) | Native Monarch endpoint calls: `--worker-comm-backend monarch` | `monarch` | `mooncake` |
| 3 (optional) | Monarch RDMA as an object-store / weight-transfer data plane | any | `monarch_rdma` |

Phase 1 is what makes the backend usable. Phases 2 and 3 are performance and ergonomics work, and are
only worth doing once phase 1 is stable.

## 2. Goals and non-goals

**Goals**

- A user picks the backend with one flag: `--cluster-backend monarch`. The training script, specs and
  flags stay the same, as they already do for Kubernetes.
- Same placement as Ray: the same GPU ordering, colocation, `pin_to_head` behaviour, and
  `--deploy-component` splits where they apply.
- Same failure semantics as the Ray worker manager: a cell that loses a worker is torn down and
  reported as not alive. Recovery stays in the cell reconcile loop.
- `torchmonarch` is an optional dependency. A Ray-only install never imports it.
- No duplicated pool/cell bookkeeping (see the [SSOT rule](../../.claude/rules/single-source-of-truth.md)).

**Non-goals (for this proposal)**

- Removing or deprecating Ray. Ray stays the default.
- Monarch's tensor engine (`monarch.fetch_shard`, `RemoteProcessGroup`). Megatron/FSDP keep their own
  `torch.distributed` setup.
- The miles dashboard and the Ray Prometheus collector. They are rejected under `monarch`, as they are
  under `kubernetes` today.
- Hot restart (`--hot-restart`). It is Kubernetes-only today, and Monarch's ownership model has no
  detached actors to restart against (§4.3).

## 3. Background

### 3.1 The three axes miles already has

| Axis | Flag | Values today | Owner |
|---|---|---|---|
| Who creates worker processes | `--cluster-backend` | `ray`, `kubernetes` | `miles/utils/workers/types.py:ClusterBackend` |
| How the driver calls a worker | `--worker-comm-backend` | `ray`, `rpc` | `types.py:WorkerCommBackend`, allowed pairs in `_SUPPORTED_WORKER_COMM_BACKENDS` |
| How rollout/train data moves | `--object-store-backend` | `ray`, `mooncake` | `miles/utils/object_store_config.py:ObjectStoreBackend` |

Kubernetes already runs with `rpc` + `mooncake` and no Ray at all, and `arguments.py` checks that
combination. That is the template Monarch phase 1 follows.

### 3.2 Where Ray actually is

`tests/fast/utils/workers/test_ray_communication_boundary.py` keeps a ledger of every module allowed to
import Ray. Grouped:

| Group | Modules | What Monarch needs |
|---|---|---|
| **Launcher closure** (process creation) | `ray_worker_manager.py`, `placement_group.py`, `wiring.py`, `worker_provider/ray.py`, `backend_capability/ray.py`, `cell_operations/ray.py`, `ray_worker_handle.py` | A Monarch counterpart for each one. The bulk of the work. |
| **Node probes** | `misc.py:get_current_node_ip`, `train_actor.py:get_local_gpu_id`, `weight_update/protocols/broadcast.py`, `transports/mooncake.py`, `object_store_config.py:_local_hostname` | Backend-neutral helpers. Most of these call `ray._private.services.get_node_ip_address()` only to learn the local IP. |
| **Launcher scripts** | `command_utils/ray_backend/*`, `external_utils/ray_job.py` | A `MonarchCommandBackend`. |
| **Ray-only features** | dashboard, `prometheus_utils.py`, `http_utils.py` distributed post, multi-LoRA (drives `RayWorkerManager`) | Rejected under `monarch`, as under `kubernetes`. |
| **Ray data plane** | `object_store.py` (`RayObjectStore`) | Replaced by `mooncake` in phase 1. |

### 3.3 What `RayWorkerManager` does

`miles/utils/workers/ray_worker_manager.py` (602 lines) is the piece Monarch replaces. Its job splits
into two halves:

1. **Backend-neutral bookkeeping.** These are already independent of Ray in substance:
   - `_PoolManager` and `_CellManager`, with generations and a membership lock.
   - `start_cells`/`stop_cells`, with rollback when a start fails.
   - The liveness scan.
   - `get_cell_infos`/`get_worker_infos`/`get_worker_addrs`.
   - Fault-hook routing.
   - Port allocation policy: static vs dynamic ports, `master` ports only on worker 0, `offset_by_cell`.
   - Choosing between the command, serve-over-RPC and serve-over-Ray paths.
2. **Ray mechanics:**
   - `ray.remote(...).options(num_gpus, scheduling_strategy=PlacementGroupSchedulingStrategy, runtime_env={"env_vars": ...})`.
   - `ray.kill`.
   - `__ray_ready__` probes.
   - `ray.method(concurrency_group=...)`.
   - The named-actor lookup `ray.get_actor("ray_worker_manager")`, which workers use to build their `BackendCapability`.

## 4. Monarch in one page

Facts below come from the Monarch repo at `c3c1447d` (2026-10-27, `torchmonarch 0.8.0.dev`, "Beta").

- **Processes and actors.** A `HostMesh` comes from a job or from `attach_to_workers(...)`.
  - `host_mesh.spawn_procs(per_host=..., bootstrap_command=..., bootstrap=...)` returns a `ProcMesh`.
  - `proc_mesh.spawn(name, ActorCls, ...)` returns an `ActorMesh`.
  - Every mesh can be sliced by named dimension.
- **Calls.** `@endpoint` methods are called with `.call()`, `.call_one()`, `.choose()`, `.stream()` or
  `.broadcast()`. They return a `Future` that can be awaited or read with `.get()`.
  - An actor's endpoints must be all sync or all async.
  - Messages from one sender are handled in FIFO order.
  - `@concurrent_endpoint` lets an async endpoint overlap with others.
- **Resources.** There are no placement groups, bundles, custom resources or node affinity.
  **Monarch does not assign GPUs.** The caller pins them by setting `CUDA_VISIBLE_DEVICES` per
  coordinate through `BootstrapCommand.with_env(...)`. Colocation (two proc meshes on the same devices)
  is explicitly supported.
- **Environment.** A proc inherits the *host agent's* environment, not the controller's. Per-proc env
  goes through `bootstrap_command`. `bootstrap=fn` runs in each proc before any actor starts.
- **Ownership and supervision.**
  - Every mesh is owned by the actor that spawned it and cannot outlive it. There are no detached actors.
  - Failures go to the owner's `__supervise__(failure)`. If nothing handles them, the root exits.
  - There is no automatic restart.
- **Discovery.** There is no named-actor registry. The closest thing is `get_or_spawn_controller(name, Cls)`,
  a per-name singleton on the root client. Mesh handles can be pickled and passed as arguments.
- **Data.** There is no object store. Messages are always pickled. `monarch.rdma.RDMABuffer` exposes
  1-D contiguous CPU or GPU memory for one-sided `read_into`/`write_from` (ibverbs, optional TCP
  fallback).
- **Bring-up.**
  - Workers run `run_worker_loop_forever(address=...)` and the controller calls `attach_to_workers(...)`.
  - Alternatively, a `SlurmJob` / `KubernetesJob` / `LocalJob` can be applied to get a `HostMesh`.
  - There is no `ray job submit`. The controller *is* the driver script.
- **Install.** `pip install torchmonarch` (Python 3.10–3.13, Linux). The wheel does not pin torch.
  Building from source needs nightly Rust.

### 4.1 Concept mapping

| Miles concept (Ray today) | Monarch equivalent | Gap to close in miles |
|---|---|---|
| `ray start` on every node + `ray job submit` | `run_worker_loop_forever` on every node (or a `SlurmJob`), driver runs `train.py` directly | New `MonarchCommandBackend` (§5.6) |
| Placement group of `{"GPU":1,"CPU":1}` bundles, sorted by node IP then GPU id | None. Use the host mesh plus an explicit `(host, gpu)` slot list. | `MonarchGpuLayout` (§5.3) |
| Fractional `num_gpus` for colocation | Two procs given the same `CUDA_VISIBLE_DEVICES` | Explicit GPU policy (§5.3) |
| `RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES` for engines | Leave `CUDA_VISIBLE_DEVICES` unset for that proc | Neutral scheduling field (§5.3) |
| `runtime_env={"env_vars": ...}` | `BootstrapCommand.with_env(...)` | none |
| `pin_to_head` | Spawn on `this_host()` (the driver's host) | none |
| `ray.kill(actor)` | `proc_mesh.stop()` | none |
| `__ray_ready__` liveness probe | Supervision events (`__supervise__`) plus a `ping` endpoint with a timeout | none |
| `ray.method(concurrency_group=...)` | No direct equivalent. Irrelevant in phase 1 (RPC server). | Phase 2 only (§6) |
| Named actor `ray_worker_manager` + `ray.get_actor` | Inject the manager mesh handle into each worker, or `get_or_spawn_controller` | §5.4 |
| `ray.put`/`ray.get` object store | None (Mooncake in phase 1, RDMA in phase 3) | §7 |
| `ray.get_gpu_ids()` | The manager knows the assignment and passes it in | §5.5 |

### 4.2 Why this layering and not "Monarch everywhere"

Monarch is a single-controller actor runtime, close to Ray core. A native port (every worker a
Monarch actor, every call an endpoint) would touch every `BaseWorkerHandle` call site. It would also
copy what the RPC comm backend already does without Ray. Treating Monarch as a **cluster backend
first** keeps the change inside `miles/utils/workers/` and the launcher. Kubernetes already proved
that boundary. Native endpoints stay available as phase 2.

### 4.3 Things Monarch does differently that matter

- **Ownership instead of detachment.** The manager owns every proc mesh. If the manager's process
  dies, every worker dies with it. That matches the current Ray behaviour in practice: the manager is
  not detached and is killed at shutdown. It does rule out hot restart.
- **No GPU assignment.** Miles must compute and set `CUDA_VISIBLE_DEVICES` itself. That is more code,
  but it is also more explicit than placement-group bundle reordering.
- **API churn.** Monarch states it does not keep backwards compatibility in its internals. Miles must
  import only `monarch.actor`, `monarch.job` and `monarch.rdma`, never `monarch._src`, and pin a
  version range.

## 5. Phase 0 + 1 design

### 5.1 User-facing surface

```bash
# single node: driver and workers on this host
python train.py --cluster-backend monarch ...            # worker-comm defaults to rpc, object store forced to mooncake

# multi node: point the driver at Monarch worker loops already running on each node
MILES_MONARCH_WORKERS=tcp://node0:26600,tcp://node1:26600 python train.py --cluster-backend monarch ...

# through the launcher, as for ray / kubernetes
MILES_SCRIPT_CLUSTER_BACKEND=monarch python scripts/run_qwen3_4b.py train
```

Changes to `types.py`:

```python
class ClusterBackend(Enum):
    RAY = "ray"
    KUBERNETES = "kubernetes"
    MONARCH = "monarch"

_SUPPORTED_WORKER_COMM_BACKENDS = {
    ClusterBackend.RAY: (WorkerCommBackend.RAY, WorkerCommBackend.RPC),
    ClusterBackend.KUBERNETES: (WorkerCommBackend.RPC,),
    ClusterBackend.MONARCH: (WorkerCommBackend.RPC,),   # + MONARCH in phase 2
}
```

**One owner for "this backend has no Ray cluster".** `arguments.py` currently hard-codes
`ClusterBackend.KUBERNETES` in several checks: rejecting the dashboard, distributed post and multi-LoRA,
and forcing `mooncake`. The same condition also appears in `tracking_utils/base.py`,
`mini_ft_controller.py` and the fault injector. Adding a third `== KUBERNETES` / `!= KUBERNETES` at each
site would violate SSOT. Instead, add derived predicates on the enum and switch every site to them:

```python
class ClusterBackend(Enum):
    ...
    def provides_ray_cluster(self) -> bool:        # dashboard, prometheus, distributed post, RayObjectStore
        return self is ClusterBackend.RAY

    def launches_workers_from_driver(self) -> bool: # RayWorkerManager-style manager exists; multi-LoRA OK
        return self in (ClusterBackend.RAY, ClusterBackend.MONARCH)

    def auto_resumes_cells(self) -> bool:           # mini_ft_controller / fault injector
        return self is ClusterBackend.KUBERNETES
```

Each site then reads the predicate that matches its *reason*. Today the reasons are mixed together
under "is kubernetes". Multi-LoRA, for example, only needs a driver-launched manager. If the Monarch
manager exposes the same verbs, it can work under `monarch`; until it is verified, keep it rejected.

`--deploy-component` splits need one cluster per deployment, and the RPC comm backend allows that
under Ray today. Monarch gets the same rule. Allow `monarch` in `_validate_deploy_component` together
with `rpc`.

### 5.2 Phase 0: split the worker manager (no behaviour change)

Copying `RayWorkerManager` and swapping Ray calls for Monarch calls would duplicate all of the
pool/cell/generation/liveness/port logic, and the two copies would drift. Instead, split it into
`miles/utils/workers/worker_manager/`:

```text
worker_manager/
├── core.py          # WorkerManager, _PoolManager, _CellManager, _BaseActorManager subclasses: all bookkeeping
├── runtime.py       # WorkerRuntime / WorkerProcess protocols (below)
├── ray_runtime.py   # PG scheduling, ray.remote options, ray.kill, __ray_ready__, concurrency-group wrapping
└── monarch_runtime.py  (phase 1)
```

The seam between them is the set of things `_BaseActorManager` does to "an actor":

```python
class WorkerProcess(Protocol):
    """One launched worker; what the core manager needs from it, whatever runs it."""
    async def call(self, method: str, /, **kwargs) -> Any: ...       # _get_node_ip, run, start_rpc_server, ...
    def call_without_result(self, method: str, /, **kwargs) -> None: ...  # CommandActor.run is fire-and-forget
    async def probe_is_dead(self) -> bool: ...
    async def kill(self) -> None: ...

class WorkerRuntime(Protocol):
    def launch(self, *, actor_class: type, ctor_kwargs: dict, placement: WorkerPlacement,
               env_vars: dict[str, str], concurrency_groups: dict[str, int] | None) -> WorkerProcess: ...
```

`WorkerPlacement` is the backend-neutral result of the current `gpu_slot_index` arithmetic: the
global slot indices, `num_cpus`, `num_gpus` and `pin_to_head`. The Ray runtime turns it into a
`PlacementGroupSchedulingStrategy`. The Monarch runtime turns it into a `(host, gpu ids)` pair.

Other pieces in this phase:

- `PortAllocator.alloc` takes a `WorkerProcess` instead of a Ray handle.
- `RayWorkerManager.launch`/`get_handle` remain as the thin Ray wrapper around the core.
- Both are mechanical moves. Do them with the
  [mechanical-refactor-verify](../../.claude/skills/mechanical-refactor-verify) workflow (prepare,
  move, postpare). The existing `tests/fast/utils/workers/test_ray_worker_manager*.py` and `real_ray/`
  suites must pass without edits.

**The client side gets the same treatment.** `RayWorkerProvider` and `RayCellOperations` are thin
wrappers that call `self._worker_manager_handle.<verb>.remote(...)`. Replace the `.remote` plumbing
with a small `WorkerManagerClient` protocol. The verbs are `get_cell_infos`, `get_worker_infos`,
`get_worker_addrs`, `get_actor_handle`, `stop_cells`, `start_cells`, `observe_fault_target` and
`control_fault_hook`. Then rename the provider and operations to `ManagerWorkerProvider` /
`ManagerCellOperations`. Ray supplies a client that calls `.remote`, Monarch one that calls
`.call_one()`. Polling, reconcile and handle construction are then written once.

### 5.3 Placement: GPU layout without placement groups

`placement_group.py:_get_placement_group_layout(args)` already computes `(num_gpus, rollout_offset)`
without depending on Ray. It handles colocate, `debug_*_only`, external rollout and deploy-component
selection. **Reuse it unchanged.** Move it, together with `_compute_trainer_num_gpus`, into a
backend-neutral `miles/utils/workers/gpu_layout.py` so the Monarch path does not import a Ray module.

The Monarch equivalent of `create_placement_groups(args)`:

```python
@dataclass(frozen=True)
class GpuSlot:
    host_rank: int      # index into the sorted host mesh
    host_ip: str
    gpu_id: int         # physical device index on that host

@dataclass(frozen=True)
class MonarchGpuLayout:          # same role as dict[str, PlacementGroupInfo]
    slots_of_pg: dict[str, list[GpuSlot]]   # "actor", "rollout", ("critic" -> same list as "actor")
```

- **PACK order.** Probe each host's IP once, sort hosts with the existing `sort_key` (IP, then GPU id),
  then fill GPUs host by host. This gives the same logical-slot to physical-GPU mapping Ray produces
  after `pg_reordered_*`. Fail fast if `num_gpus > hosts × num_gpus_per_node`.
- **Slicing.** `"rollout"` is `slots[rollout_offset:]`, exactly as with PGs, so
  `SchedulingSpec.pg_name`/`pg_slot_offset` keep their meaning.
- **Colocation.** With `--colocate`, actor and rollout slots overlap on the same physical GPUs. Monarch
  allows two procs on one device, so no change is needed beyond the GPU policy below.

**GPU visibility policy.** Ray hides this behind fractional `num_gpus` plus `NOSET` env vars. Monarch
needs it stated explicitly:

| Worker kind | Ray today | Monarch |
|---|---|---|
| Trainer rank (Megatron/FSDP), GPU slots > 0 | Ray sets `CUDA_VISIBLE_DEVICES` to its bundle GPU | Set `CUDA_VISIBLE_DEVICES=<its physical ids>` in `bootstrap_command` |
| SGLang engine (`InferenceEngineSpec`) | `RAY_EXPERIMENTAL_NOSET_*` so it sees the whole node and picks `base_gpu_id` | Leave `CUDA_VISIBLE_DEVICES` unset (sees the whole host) |
| CPU-only workers (controllers, router, rollout executor, session server) | `num_gpus=0` | `CUDA_VISIBLE_DEVICES=""` |

"Sees the whole node" is currently stored as Ray-specific env var names inside
`compute_inference_engine_env_vars`. Make it a neutral `SchedulingSpec` field, e.g.
`sees_all_node_gpus: bool`. The Ray runtime translates it into the `NOSET` env vars, and the Monarch
runtime into "don't set `CUDA_VISIBLE_DEVICES`". The decision then has one owner, the spec, and the
`NOSET` list stays inside the Ray runtime.

**Proc granularity.** Use **one single-proc `ProcMesh` per worker**: `host_mesh.slice(hosts=h)`,
then `.spawn_procs(per_host={"procs": 1}, bootstrap_command=...)`. A cell's workers can be spread
unevenly across hosts, for example a rollout offset that is not aligned to a host boundary, so a cell
is not always a rectangular mesh. Per-worker meshes give exact parity with Ray's per-bundle actors.
They also keep `stop`/`probe` per worker, which is what `_CellManager` expects. One `ProcMesh` per
cell is a later optimisation for the common aligned case.

### 5.4 The manager process and how workers find it

- **Where it runs.** The driver spawns `MonarchWorkerManager` (a Monarch `Actor` with async endpoints
  wrapping the core manager) in **its own proc on the driver host**:
  `this_host().spawn_procs(per_host={"procs": 1})`. This matches Ray, where the manager is a separate
  actor with its own event loop. It also means the manager, as owner of every worker mesh, receives
  `__supervise__` events.
- **Discovery.** Workers build a `DeferredBackendCapability`, and under Ray that calls
  `RayWorkerManager.get_handle()` (`ray.get_actor`). Monarch has no registry. The recommended fix is
  **handle injection**:
  - The manager passes its own actor handle (`context().actor_instance` as a mesh ref; it pickles)
    into each worker's wrapper actor constructor.
  - The wrapper stores it in a process-global the Monarch capability factory reads.
  - This is explicit, needs no global name, and survives proc restarts because each launch injects it
    again.
  - Fallback: `get_or_spawn_controller("miles_worker_manager", ...)`, if it turns out to work from
    non-root procs. That needs checking (§9).
- **Capability.** `factory.get_backend_capability` gains a `ClusterBackend.MONARCH` case that returns
  the generic `ManagerBackendCapability` over a `MonarchWorkerManagerClient`. Imports must be **lazy**:
  `factory.py` imports Ray modules eagerly today, so split it so that each backend branch imports
  its own module.

### 5.5 Worker wrappers, env and node probes

**Wrapper actors.** The core manager launches three plain Python classes: `CommandActor`,
`ServeActor`, and `bootstrapped_worker_class(...)` for the Ray comm path. Monarch needs `Actor`
subclasses with `@endpoint`s. Add one generic adapter:

```python
class MonarchWorkerActor(Actor):
    def __init__(self, *, target_factory, manager_handle): ...
    @endpoint
    def invoke(self, method: str, kwargs: dict) -> Any:  # sync endpoints: CommandActor/ServeActor are sync
        return getattr(self._target, method)(**kwargs)
    @endpoint
    def ping(self) -> bool: return True
```

`WorkerProcess.call(method, **kw)` maps to `actor.invoke.call_one(method, kw)`. A single `invoke`
endpoint avoids generating a Monarch class per worker class. In phase 1 the real worker API is served
over HTTP RPC anyway, so `invoke` only carries launcher verbs: `_get_node_ip`, `_get_free_port_block`,
`run`, `start_rpc_server`, `shutdown` and `control_fault_hook`.

**Env.** `spec.env_var(launch_context)` already returns a plain dict, and `with_env(...)` consumes it
directly. The launcher also forwards `MASTER_ADDR`, `NCCL_*`, `PYTHONPATH` and `no_proxy`. Under Ray
those travel through the job's `runtime_env`. Under Monarch, procs inherit the host agent's env, so
these must either be set where `run_worker_loop_forever` starts, or be added explicitly with
`with_env` from a single `compute_run_env_vars()` the launcher already builds (`train_env_vars`).
Prefer the explicit route: one source, no reliance on how each node was started.

**Node probes.** These replace the remaining Ray calls on the worker side:

- `misc.get_current_node_ip()`: keep `MILES_HOST_IP_ENV` as the override. Fall back to a Ray-free
  implementation (the UDP-connect trick Ray itself uses) instead of
  `ray._private.services.get_node_ip_address()`. Then point `broadcast.py`, `transports/mooncake.py`
  and `object_store_config._local_hostname` at `get_current_node_ip()`. **One owner** for "my IP",
  and four entries leave the Ray ledger.
- `train_actor.get_local_gpu_id()`: under Monarch the manager has set `CUDA_VISIBLE_DEVICES` to the
  rank's physical ids, so the local id is `0`. To avoid a backend `if`, have the manager set
  `LOCAL_RANK` (or a `MILES_LOCAL_GPU_ID`) in `env_var` for every backend. The function then reads
  that and keeps `ray.get_gpu_ids()` only as the Ray fallback.

### 5.6 Liveness and failure

There are two signals. Both end in the existing `_CellManager` teardown path:

1. **Supervision (push).** `MonarchWorkerManager.__supervise__(failure)`:
   - Map `failure` to its proc mesh, then to its cell, and schedule the same "lost workers, tear down
     the cell" path `_scan_liveness_once` uses (generation-checked, under `_membership_lock`).
   - Return `True` (handled) so the failure does not escalate to the root and kill the run.
   - Monarch may call `__supervise__` 1 to N times for N failing actors, so the handler must be
     idempotent. The generation check already makes it so.
2. **Probe (pull).** `WorkerProcess.probe_is_dead()` calls `ping` with a short timeout.
   - A `SupervisionError` or a stopped mesh means dead.
   - A timeout means inconclusive, same as Ray's `ActorUnavailableError`.
   - The 10 s scan remains as a backstop.

`ServeActor` and `CommandActor` call `os._exit` when their server or subprocess dies. Under Monarch
that kills the proc, so supervision fires: the same behaviour as Ray, with faster detection.

### 5.7 Launcher: `MonarchCommandBackend`

Add `command_utils/monarch_backend/backend.py` next to `ray_backend/` and `helm_backend/`, and select
it in `CommandUtilConfig.create_backend()`.

| Step | Ray backend | Monarch backend |
|---|---|---|
| Clean up | `ray stop`, `pkill` | `pkill` of the worker loop and sglang |
| Bring up | `ray start --head` | Start `python -m miles.utils.workers.monarch_worker_loop --port P` on each node (it wraps `run_worker_loop_forever`) |
| Multi-node exec (`prepare_cmd`, `exec_command_multi_node`) | Ray tasks with `NodeAffinitySchedulingStrategy` | A one-shot `ShellActor` spawned per host on the attached `HostMesh` |
| Mooncake master | `start_mooncake_master` | Same function, unchanged |
| Submit | `ray job submit -- python3 train.py ...` | Run `python3 train.py ... --cluster-backend monarch` directly with `MILES_MONARCH_WORKERS` set. The driver is the Monarch client. |

**Host acquisition** gets one owner: `miles/utils/workers/monarch_hosts.py:acquire_host_mesh(args)`,
resolved in this order:

1. `MILES_MONARCH_WORKERS` (comma-separated `tcp://host:port`) → `attach_to_workers(...)`.
2. Otherwise, under SLURM (`SLURM_JOB_NODELIST`), derive those addresses from the allocation, since
   the launcher already reads `SLURM_JOB_NUM_NODES`.
3. Otherwise, single node: `this_host()`.

Letting miles itself submit a `SlurmJob`/`KubernetesJob` is deliberately left out of phase 1. Today
`ray start` is also outside miles' scheduling responsibility.

### 5.8 Feature matrix after phase 1

| Feature | `ray` | `monarch` (phase 1) | Notes |
|---|---|---|---|
| Sync / async training, colocate, offload | ✅ | ✅ | via `rpc` |
| Megatron / FSDP / torchtitan trainers | ✅ | ✅ | trainers do their own `torch.distributed` setup |
| Weight update (NCCL broadcast, p2p, Mooncake transport) | ✅ | ✅ | after the node-IP fix (§5.5) |
| `--object-store-backend ray` | ✅ | ❌ → forced `mooncake` | as under `kubernetes` |
| Cell fault tolerance (`--use-fault-tolerance`) | ✅ | ✅ | supervision + probe |
| `--deploy-component` splits | ✅ (with `rpc`) | ✅ (with `rpc`) | one Monarch client per deployment |
| Dashboard, Prometheus collector, distributed post | ✅ | ❌ | `provides_ray_cluster()` |
| Multi-LoRA | ✅ | ⚠️ rejected until verified | needs only the manager verbs |
| Hot restart | ❌ | ❌ | K8s only |
| TPU (`MILES_ACCELERATOR=tpu`) | ✅ | ❌ initially | Monarch GPU pinning is CUDA-env based; gate in the platform |

## 6. Phase 2 (optional): `--worker-comm-backend monarch`

Phase 2 adds a `MonarchWorkerHandle(BaseWorkerHandle)`, the counterpart of `RayWorkerHandle`. Calls go
straight to Monarch endpoints instead of HTTP.

- **Wins:** no per-worker HTTP server or port, FIFO ordering per sender, and pickled payloads without
  JSON/HTTP framing.
- **Costs:**
  - Every served worker class needs an `Actor` wrapper. Generate it the way
    `_ServeActorRayCommManager._compute_actor_class` does today.
  - `concurrency_groups` have no Monarch equivalent. A worker with groups (the trainer under
    `--use-fault-tolerance`) needs async endpoints plus `@concurrent_endpoint`, or must stay on `rpc`.
  - The sync/async-endpoint purity rule must be checked per class.
- **Gate:** phase 2 only when profiling shows RPC overhead matters. The RPC path already works and is
  shared with Kubernetes.

## 7. Phase 3 (optional): Monarch RDMA data plane

- **Object store:** `MonarchRdmaObjectStore(BaseObjectStore)`.
  - `put` pins the serialized buffer in the producing process and returns an `RDMABuffer` handle (it
    pickles) inside a `_MonarchStoreObjectRef`.
  - `get` does `read_into`.
  - `remove` does `drop()`.
  - Add `ObjectStoreBackend.MONARCH_RDMA` to the discriminated `StoreObjectRef` union.
  - Lifetime is tied to the producer process, which is the same weakness as Ray's owner-based refs.
- **Weight transfer:** a transport next to `transports/mooncake.py`, following the learner-exposes-
  `RDMABuffer`s / engine-pulls pattern from Monarch's GRPO example. Worth doing only if it beats the
  existing p2p/Mooncake transports on the target fabric.

## 8. Testing and CI

- **Boundary ledger.** Add `test_monarch_communication_boundary.py`, which mirrors the Ray ledger with
  a `MONARCH_USING_MODULES` dict (path → reason). It keeps `torchmonarch` imports inside
  `monarch_runtime.py`, `monarch_hosts.py`, the Monarch capability/client and the launcher. Also
  assert that `import miles` works without `torchmonarch` installed.
- **Phase 0 safety net.** The existing `test_ray_worker_manager*.py` and `real_ray/` suites, unchanged.
- **Fast tests (CPU):**
  - Core manager tests run against a `FakeWorkerRuntime`, as `fake_ray.py` does for Ray.
  - `MonarchGpuLayout` tests check slot ordering, colocate overlap, rollout offset and the
    insufficient-hosts error.
  - Predicate tests check the `ClusterBackend` predicates and the argument validation matrix.
- **Real Monarch, CPU.** Monarch runs on CPU with local procs, so `tests/fast/utils/workers/real_monarch/`
  can:
  - run the existing `conformance.py` suite (it is backend-parametrised already) against
    `--cluster-backend monarch`;
  - test the supervision path by killing a worker proc;
  - test handle injection.
- **E2E (GPU).** One small colocated GRPO run and one disaggregated run under `--cluster-backend monarch`,
  registered per [`ci-test-registration.md`](../../.claude/rules/ci-test-registration.md). Start them
  `disabled=` until the image ships `torchmonarch`.
- **Image.** Add `torchmonarch` (a pinned range) as an optional extra (`pip install miles[monarch]`) and
  to the dev Docker image. Check its torch ABI against the image's torch on every bump.

## 9. Risks and open questions

| # | Item | Mitigation / decision needed |
|---|---|---|
| 1 | Monarch is Beta and its internals change without notice | Import only public `monarch.actor`/`job`/`rdma`; pin a range; one adapter module absorbs churn |
| 2 | Does `get_or_spawn_controller` resolve from non-root procs? | Not needed if we go with handle injection (recommended). Verify before choosing the fallback. |
| 3 | SGLang with no `CUDA_VISIBLE_DEVICES` and `base_gpu_id` under colocation | Same contract as Ray `NOSET` today; covered by the colocated e2e |
| 4 | Host agents inherit their own env, not the driver's | Explicit `with_env` from `train_env_vars` (§5.5); document how worker loops must be started |
| 5 | `__supervise__` called 1..N times; escalation kills the root | Idempotent generation-checked handler that always returns handled for worker meshes |
| 6 | Monarch fork rules (WF-1..5) vs `CommandActor` spawning subprocesses | `CommandActor` uses `Popen` (fork+exec), likely fine; test in `real_monarch` |
| 7 | Monarch's own ports and miles' dynamic port range (20000+) | Pin the worker-loop port and Monarch's listen range outside `addr_allocator._DYNAMIC_PORT_START..` |
| 8 | Multi-LoRA under `monarch` | Enable once the manager verbs it drives are covered by a test |
| 9 | Naming | Module and class names follow [pool-cell-worker-names](../../.claude/rules/pool-cell-worker-names.md): no `fleet`/`group` in new identifiers |

## 10. Delivery plan (PR sequence)

1. **Predicates.** Add `ClusterBackend` predicates and switch the existing `== KUBERNETES` sites to them.
   No behaviour change.
2. **Node probes.** A Ray-free `get_current_node_ip`, with every site pointed at it. `LOCAL_RANK` is
   set by the manager. Four modules leave the Ray ledger.
3. **Neutral GPU visibility.** Add `SchedulingSpec.sees_all_node_gpus` and move the `NOSET` list into
   the Ray runtime.
4. **Layout move.** Move `gpu_layout.py` (the layout math) out of `placement_group.py`.
5. **Phase 0 split.** `worker_manager/{core,runtime,ray_runtime}.py` plus `WorkerManagerClient` and
   the generic provider/cell operations. Verified as a mechanical refactor.
6. **Monarch runtime.** `ClusterBackend.MONARCH`, `monarch_runtime.py`, `MonarchGpuLayout`, handle
   injection, supervision, the `real_monarch` CPU tests and the boundary ledger.
7. **Launcher and docs.** `MonarchCommandBackend`, `monarch_hosts.py`, the worker-loop entrypoint,
   e2e registration, and a "Monarch" section in [Ray and Kubernetes Backend](/advanced/cluster-backend).
8. **Later.** Phase 2 and phase 3 behind their own flags, each justified by measurements.

PRs 1–5 improve the codebase even if Monarch never lands. Each one removes a Ray assumption from code
that is supposed to be backend-neutral already.
