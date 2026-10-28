#!/bin/bash
# Build the aarch64 + CUDA 13 venv the Monarch e2e runs used, from PyPI alone (no docker image, no network on
# the compute nodes: run this on a login node, into a directory the compute nodes mount).
#
#   E2E_ROOT=~/miles_e2e PYTHON=<python3.12 with headers> bash tests/manual/monarch_e2e/build_env.sh
#
# PYTHON must ship Python.h (triton compiles a C stub at runtime); a uv-managed CPython does, a distro python
# without -dev headers does not.
set -ex
E2E_ROOT=${E2E_ROOT:-$HOME/miles_e2e}
PYTHON=${PYTHON:?point PYTHON at a python3.12 that ships its headers, e.g. a uv-managed cpython}
MILES_SRC=$(cd "$(dirname "$0")/../../.." && pwd)
export UV_CACHE_DIR=$E2E_ROOT/uv_cache
mkdir -p "$E2E_ROOT"
uv venv --python "$PYTHON" "$E2E_ROOT/venv"
source "$E2E_ROOT/venv/bin/activate"

# Stock sglang: the sglang-miles fork the docker image overlays was not installable here, see the README.
uv pip install "sglang==0.5.22" "torch==2.14.1" -r "$MILES_SRC/requirements.txt" \
  "torchmonarch==0.8.0.dev20261028" \
  mooncake-transfer-engine-cuda13 \
  megatron-core \
  pytest pytest-timeout
# The JIT kernels need a compiler whose PTX the 13.0 runtime headers and ptxas agree with.
uv pip install "torch==2.14.1" "nvidia-cuda-nvcc==13.0.88" "nvidia-cuda-crt==13.0.88" "nvidia-nvvm==13.0.88" \
  "nvidia-cuda-cccl==13.0.85"

# A CUDA_HOME with the unversioned libraries the JIT linkers ask for (-lcudart), over the pip CUDA tree.
CU13=$E2E_ROOT/venv/lib/python3.12/site-packages/nvidia/cu13
CUDA_HOME=$E2E_ROOT/cuda_home
rm -rf "$CUDA_HOME" && mkdir -p "$CUDA_HOME/lib64"
for d in bin include nvvm cccl; do ln -s "$CU13/$d" "$CUDA_HOME/$d"; done
for f in "$CU13"/lib/*; do
  b=$(basename "$f"); ln -s "$f" "$CUDA_HOME/lib64/$b"
  [[ $b == *.so.* ]] && [ ! -e "$CUDA_HOME/lib64/${b%%.so*}.so" ] && ln -s "$f" "$CUDA_HOME/lib64/${b%%.so*}.so"
done
ln -s "$CUDA_HOME/lib64" "$CUDA_HOME/lib"

cat > "$E2E_ROOT/env.sh" <<ENV
source $E2E_ROOT/venv/bin/activate
export CUDA_HOME=$CUDA_HOME
export PATH=\$CUDA_HOME/bin:\$PATH
ENV
python -c "import torch, sglang, monarch; print(torch.__version__, sglang.__version__)"
