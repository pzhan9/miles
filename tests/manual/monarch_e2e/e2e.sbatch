#!/bin/bash
#SBATCH -A sysml
#SBATCH -p g3
#SBATCH --qos=g3_lowest
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH -c 32
#SBATCH --mem=256G
#SBATCH -t 60
# One 10-step GRPO run of Qwen3-0.6B (FSDP) on two gpus; account/partition above are the ones the reported runs
# used, override them on the sbatch command line.
#
#   MILES_SRC=<checkout> [LAYOUT=disagg] [START_MOONCAKE=1] [SAMPLE_USAGE=1] \
#     sbatch --export=ALL -J <name> -o <log> tests/manual/monarch_e2e/e2e.sbatch <label> <extra train args...>
#
# LAYOUT=disagg puts the trainer and the engine on one gpu each (default: both colocated on both gpus).
# START_MOONCAKE=1 starts a Mooncake master for runs that use --object-store-backend mooncake.
# SAMPLE_USAGE=1 samples gpu processes and the top cpu users every 10 s into the run directory.
set -x
LABEL=$1; shift
E2E_ROOT=${E2E_ROOT:-$HOME/miles_e2e}
MODEL=${MODEL:-/checkpoint/pytorch/shared/Qwen3-0.6B}
DATA=${DATA:-$E2E_ROOT/data/arith.jsonl}
source "$E2E_ROOT/env.sh"
cd "${MILES_SRC:-$SLURM_SUBMIT_DIR}"
git log -1 --format="%h %s"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export MASTER_ADDR=$(hostname -i | awk '{print $1}')
export no_proxy="127.0.0.1,localhost,$MASTER_ADDR" NO_PROXY="127.0.0.1,localhost,$MASTER_ADDR"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
OUT=$E2E_ROOT/runs/$LABEL-$SLURM_JOB_ID
mkdir -p "$OUT"
nvidia-smi -L

if [ "$LAYOUT" = disagg ]; then
  LAYOUT_ARGS="--actor-num-gpus-per-node 1 --rollout-num-gpus 1"
else
  LAYOUT_ARGS="--actor-num-gpus-per-node 2 --colocate"
fi
if [ -n "$SAMPLE_USAGE" ]; then
  (while true; do
     date +%T
     nvidia-smi --query-compute-apps=pid,used_memory,gpu_bus_id --format=csv,noheader
     nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
     top -b -n1 -o %CPU | sed -n 8,22p
     sleep 10
   done > "$OUT/usage.log" 2>&1 &)
fi
if [ -n "$START_MOONCAKE" ]; then
  (setsid mooncake_master --rpc_port 50051 --metrics_port 0 > "$OUT/mooncake_master.log" 2>&1 &)
  sleep 5
fi

# The sglang flags are environment workarounds for stock sglang on GB300 (see the README), the same for every
# backend: --sglang-attention-backend triton and --sglang-sampling-backend pytorch avoid kernels that download
# cubins or need sglang-miles' toolchain.
python3 train.py \
  --hf-checkpoint "$MODEL" --ref-load "$MODEL" \
  --prompt-data "$DATA" --input-key prompt --label-key label \
  --apply-chat-template --rollout-shuffle --rm-type deepscaler \
  --num-rollout "${NUM_ROLLOUT:-10}" --rollout-batch-size 8 --n-samples-per-prompt 4 \
  --rollout-max-response-len 512 --rollout-temperature 1 --global-batch-size 32 \
  --advantage-estimator grpo --eps-clip 0.2 --eps-clip-high 0.28 \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 \
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.5 \
  --sglang-attention-backend triton --sglang-sampling-backend pytorch \
  --train-backend fsdp --attn-implementation sdpa --update-weight-buffer-size 536870912 \
  --use-dynamic-batch-size --max-tokens-per-gpu 4096 \
  --actor-num-nodes 1 --num-gpus-per-node 2 $LAYOUT_ARGS \
  "$@" 2>&1
echo "EXIT CODE $?"
pkill -f mooncake_master || true
