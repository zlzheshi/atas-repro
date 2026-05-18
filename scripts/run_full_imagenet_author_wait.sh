#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/mnt/t1b6/xuzhejia/atas-repro}
CONDA_ROOT=${CONDA_ROOT:-/mnt/t1b6/xuzhejia/app/miniconda3}
DATA_ROOT=${DATA_ROOT:-/mnt/t1b6/xuzhejia/datasets/imagenet/train}
CONFIG=${CONFIG:-configs/atas_vitb_imagenet_full_author.yaml}
GPU_LIST=${GPU_LIST:-0,1,2,3}
MEMORY_THRESHOLD_MIB=${MEMORY_THRESHOLD_MIB:-2000}
POLL_SECONDS=${POLL_SECONDS:-300}
MASTER_PORT=${MASTER_PORT:-29531}
LOG_DIR=${LOG_DIR:-/mnt/t1b6/xuzhejia/logs}

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}
if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "GPU_LIST must contain at least one GPU index." >&2
  exit 1
fi

echo "[full-author] project=$PROJECT_DIR"
echo "[full-author] data=$DATA_ROOT"
echo "[full-author] config=$CONFIG"
echo "[full-author] waiting GPUs=$GPU_LIST memory_threshold=${MEMORY_THRESHOLD_MIB}MiB"

while true; do
  ready=1
  snapshot=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  echo "[$(date '+%F %T')] GPU snapshot:"
  echo "$snapshot"

  for gpu in "${GPUS[@]}"; do
    used=$(echo "$snapshot" | awk -F', ' -v id="$gpu" '$1 == id {print $2}')
    if [[ -z "$used" || "$used" -gt "$MEMORY_THRESHOLD_MIB" ]]; then
      ready=0
    fi
  done

  if [[ "$ready" -eq 1 ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

echo "[$(date '+%F %T')] GPUs are ready. Starting author-like full ImageNet ATAS training."
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate atas

CUDA_VISIBLE_DEVICES="$GPU_LIST" PYTHONPATH=. \
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NUM_GPUS" \
  --master_port="$MASTER_PORT" \
  train_atas.py \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT"
