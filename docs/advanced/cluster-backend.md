---
title: Ray, Monarch and Kubernetes Backend
description: Run the same training script on Ray or Monarch, or let Kubernetes schedule every worker of the run.
---
Miles runs a job on one of three cluster backends. Ray is the default and needs no configuration.
Monarch launches every worker from the driver as Monarch processes instead of Ray actors. Kubernetes
installs the run as a helm release, so the cluster schedules every worker. The training script is the
same either way.

## Ray

Ray is the default backend, so `python scripts/run_*.py train` launches a run as is, with nothing to
configure. Nothing in the Kubernetes chapter below applies to it.

## Monarch

<Warning>

**Status.** Experimental. The design and its phases are in
[Design: Monarch Cluster Backend](/developer/monarch-backend-design).

</Warning>

Install the optional dependency with `pip install miles[monarch]`, then pass `--cluster-backend monarch`:

```bash
MILES_SCRIPT_CLUSTER_BACKEND=monarch python scripts/run_qwen3_4b.py train
```

- **One process mesh per cell.** The driver spawns a worker manager actor, which spawns every cell as one
  Monarch proc mesh and watches it through Monarch supervision: a cell that loses a process is torn down
  and reported as not alive, exactly as under Ray.
- **Gpu placement.** Monarch assigns no gpus, so Miles packs its gpu slots onto hosts in mesh order, as a
  Ray `PACK` placement group would, and every gpu worker selects the devices it was placed on.
- **Data plane.** There is no Ray object store, so `--object-store-backend` is forced to `mooncake`, as
  under Kubernetes; the launcher starts the Mooncake master.
- **Hosts.** A single-node run spawns everything on the driver's host. A multi-node run starts
  `python -m miles.utils.workers.monarch.worker_loop --address <proc-addr>` on every host and gives the
  driver every address, comma separated, in `MILES_MONARCH_WORKERS`.
- **Not available.** The dashboard, the Prometheus collector, `--use-distributed-post` and multi-LoRA
  need a Ray cluster and are refused; hot restart is Kubernetes-only.

## Kubernetes

<Warning>

**Status.** Under active development: flags, chart values and failure semantics still change. Ray
also runs *inside* Kubernetes, and that is the well-trodden path — take this one only if you want
Miles to create the cluster's objects itself.

</Warning>

### Launch

Run everything from the repository root, with `kubectl` and `helm` on your PATH.

Install a workbench — the long-lived pod you launch from — once per namespace, from your
cluster's [`infra.yaml`](#for-cluster-administrator):

```bash
export MILES_NS="miles-$USER"
python -m miles.utils.external_utils.miles_workbench install -n "$MILES_NS" -f infra.yaml
```

It creates the namespace if missing, checks your rights, installs, and waits for Ready. Runs
launched from the pod inherit those values.

Launch from inside it:

```bash
python -m miles.utils.external_utils.miles_workbench exec -n "$MILES_NS" -- \
  bash -lc "cd /root/miles && python scripts/run_qwen3_4b.py train"
```

The recipe defaults to `/root/models`, `/root/datasets` and `/root/shared_data`; the example
[`infra.yaml`](#for-cluster-administrator) mounts all three from the shared volume. Fill them once
with `python scripts/run_qwen3_4b.py prepare`, or pass `--model-dir`, `--data-dir` and
`--output-dir` to point the recipe elsewhere.

### Observability

**Built in**

- The launcher follows every pod's logs, prints status changes and warning events, and ends with
  the run's exit code.
- On a failure, collect the namespace's logs, describes and events into one directory — before
  cleaning up, and the whole namespace, because the explanation is usually the pod next to it:
  `python -m miles.utils.external_utils.miles_workbench collect-diagnosis -n "$MILES_NS" --output-dir ~/artifacts/miles`

**External**

- A run's pods are ordinary pods, so whatever the cluster already runs — a metrics stack, a log
  collector, the platform's own dashboards — sees them with no wiring from Miles.
- Prefer it at scale. The built-in following is meant for watching one run, not hundreds of pods.

### Clean up

```bash
python -m miles.utils.external_utils.miles_workbench stop -n "$MILES_NS" 260811-143000-042
python -m miles.utils.external_utils.miles_workbench uninstall -n "$MILES_NS"
```

`stop` removes the run and frees its GPUs; `uninstall` removes the workbench.

### Folder convention

A run is many pods on many machines, and they share nothing but the volumes `infra.yaml` mounts.

- Every path your script names — `/root/models`, `/root/datasets`, `infra.paths.runsRoot` — has to
  be under one of those mounts.
- To run your own branch, mount it at `/root/miles`, `/root/Megatron-LM` or `/sgl-workspace/sglang`;
  a copy anywhere else is not imported.

### For cluster administrator

Everything above assumes this was done once.

**Install LWS.** Miles deploys its worker pools as
[LeaderWorkerSets](https://github.com/kubernetes-sigs/lws). Install the CRDs and controller, and
grant users rights over them explicitly.

**Give each user a namespace.** The namespace is the real boundary, not the Role: anything that
may create workloads can name another ServiceAccount and read its token. Keep privileged accounts
out of it.

**Write one `infra.yaml`.** The same file drives every Miles chart:

```yaml
infra:
  image:
    repository: radixark/miles
    tag: dev
  volumes:
    - name: cluster-storage
      hostPath: {path: /cluster-storage, type: Directory}
      mounts:
        - {mountPath: /cluster-storage}
        - {mountPath: /root/miles, subPath: alice/miles}
        - {mountPath: /root/models, subPath: models}
        - {mountPath: /root/datasets, subPath: datasets}
        - {mountPath: /root/shared_data, subPath: alice/shared_data}
  paths:
    runsRoot: /cluster-storage/${NAMESPACE}/miles_data
```

The `hostPath` above stands for a cluster-wide shared filesystem (NFS, Lustre, a CSI mount) already
mounted at `/cluster-storage` on every node: a per-node directory would lose the orchestrator state
file and the shared checkpoints. Where Pod Security forbids `hostPath`, replace that key with an RWX
`persistentVolumeClaim` and keep the mounts.

Any path in `infra.yaml` — a `hostPath`, a `mountPath`, a `subPath`, `infra.paths.runsRoot` — may
name `${NAMESPACE}`, which the chart replaces with the namespace it is installed into.

`charts/miles-run/values.yaml` shows the full shape, and each chart's `values.schema.json` is the
authoritative field list.
