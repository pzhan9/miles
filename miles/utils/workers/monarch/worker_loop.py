"""Serve one host of a multi-node ``--cluster-backend monarch`` run.

Usage, on every host before the driver starts:

    python -m miles.utils.workers.monarch.worker_loop --address '<proc-id>@tcp://<this-host>:26600'

and give the driver every such address, comma separated, in ``MILES_MONARCH_WORKERS``.
"""

import argparse

from monarch.actor import run_worker_loop_forever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--address", required=True, help="The ProcAddr the driver attaches to this host by.")
    run_worker_loop_forever(ca="trust_all_connections", address=parser.parse_args().address)


if __name__ == "__main__":
    main()
