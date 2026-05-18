#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/atas_vitb_24gb.yaml}
DATA_ROOT=${2:-/data/imagenet/train}
GPUS=${GPUS:-1}

if [ "$GPUS" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="$GPUS" train_atas.py \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT"
else
  python train_atas.py \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT"
fi

