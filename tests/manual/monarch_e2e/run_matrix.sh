#!/bin/bash
# Submit the e2e runs of the Monarch backend report (docs/developer/monarch-backend-e2e-report.md).
#
#   BASE_SRC=<checkout of the commit before the backend> MONARCH_SRC=<checkout of the branch> \
#     bash tests/manual/monarch_e2e/run_matrix.sh
#
# Both checkouts need sglang_compat.patch applied when the venv carries stock sglang instead of sglang-miles.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
E2E_ROOT=${E2E_ROOT:-$HOME/miles_e2e}
BASE_SRC=${BASE_SRC:?checkout of the commit before the monarch backend}
MONARCH_SRC=${MONARCH_SRC:?checkout of the monarch branch}
mkdir -p "$E2E_ROOT/logs"
NO_GRAPH="--sglang-disable-cuda-graph"  # colocated offload needs torch_memory_saver's preload hook mode

submit() {  # name src [env...] -- train args...
  local name=$1 src=$2; shift 2
  local env=()
  while [ "$1" != "--" ]; do env+=("$1"); shift; done; shift
  env "${env[@]}" MILES_SRC="$src" E2E_ROOT="$E2E_ROOT" \
    sbatch --export=ALL -J "e2e-$name" -o "$E2E_ROOT/logs/e2e-$name-%j.log" "$HERE/e2e.sbatch" "$name" "$@"
}

submit ray-colocated "$BASE_SRC" -- $NO_GRAPH
submit monarch-p1 "$MONARCH_SRC" START_MOONCAKE=1 -- --cluster-backend monarch --object-store-backend mooncake $NO_GRAPH
submit monarch-p3-rpc "$MONARCH_SRC" -- --cluster-backend monarch $NO_GRAPH
submit monarch-p23 "$MONARCH_SRC" -- --cluster-backend monarch --worker-comm-backend monarch $NO_GRAPH
submit ray-disagg "$BASE_SRC" LAYOUT=disagg --
submit monarch-p4 "$MONARCH_SRC" LAYOUT=disagg -- \
  --cluster-backend monarch --worker-comm-backend monarch --update-weight-transfer-mode monarch_rdma
