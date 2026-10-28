"""Where a ``--cluster-backend monarch`` run finds the hosts its gpu workers run on."""

from __future__ import annotations

import logging
import os

from monarch.actor import HostMesh, attach_to_workers

logger = logging.getLogger(__name__)

# Comma separated ProcAddrs (``<proc-id>@tcp://host:port``) of Monarch worker loops already serving on every
# host of the run, as ``python -m miles.utils.workers.monarch.worker_loop`` prints them.
MONARCH_WORKERS_ENV_VAR = "MILES_MONARCH_WORKERS"


def acquire_worker_hosts(*, head_host: HostMesh) -> HostMesh:
    if not (workers := os.environ.get(MONARCH_WORKERS_ENV_VAR)):
        logger.info(f"{MONARCH_WORKERS_ENV_VAR} is unset, so every worker runs on the driver's host")
        return head_host

    addrs = [addr.strip() for addr in workers.split(",") if addr.strip()]
    logger.info(f"Attaching to the monarch worker loops at {addrs}")
    hosts = attach_to_workers(ca="trust_all_connections", workers=addrs, name="miles-workers")
    hosts.initialized.get()
    return hosts
