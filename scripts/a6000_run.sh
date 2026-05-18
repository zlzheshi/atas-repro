#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/mnt/t1b6/xuzhejia/atas-repro
CONDA_ROOT=/mnt/t1b6/xuzhejia/app/miniconda3

CONFIG=${1:-configs/atas_vitb_debug.yaml}
DATA_ROOT=${2:-/mnt/t1b6/xuzhejia/datasets/imagenet/train}
GPUS=${GPUS:-1}

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate atas
cd "$PROJECT_DIR"

echo "project: $PROJECT_DIR"
echo "config:  $CONFIG"
echo "data:    $DATA_ROOT"
echo "gpus:    ${CUDA_VISIBLE_DEVICES:-all visible} / nproc=$GPUS"

if [ "$GPUS" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="$GPUS" train_atas.py \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT"
else
  python train_atas.py \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT"
fi

