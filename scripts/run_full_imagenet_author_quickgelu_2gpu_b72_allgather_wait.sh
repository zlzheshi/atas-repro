#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/mnt/t1b6/xuzhejia/atas-repro}
CONDA_ROOT=${CONDA_ROOT:-/mnt/t1b6/xuzhejia/app/miniconda3}
DATA_ROOT=${DATA_ROOT:-/mnt/t1b6/xuzhejia/datasets/imagenet/train}
LOG_DIR=${LOG_DIR:-/mnt/t1b6/xuzhejia/logs}
CONFIG=${CONFIG:-configs/atas_vitb_imagenet_full_author_quickgelu_2gpu_b72_allgather.yaml}
GPUS=${GPUS:-0,1}
MIN_FREE_MB=${MIN_FREE_MB:-30000}
POLL_SECONDS=${POLL_SECONDS:-120}

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate atas
cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"

gpu_free_mb() {
  local gpu=$1
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | awk '{print $1}'
}

wait_for_selected_gpus() {
  while true; do
    local ready=1
    for gpu in "${GPU_LIST[@]}"; do
      local free_mb
      free_mb=$(gpu_free_mb "$gpu")
      if (( free_mb < MIN_FREE_MB )); then
        ready=0
      fi
    done
    if (( ready == 1 )); then
      return 0
    fi
    echo "[$(date '+%F %T')] waiting for GPUs ${GPUS} with >= ${MIN_FREE_MB} MiB free memory..."
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    sleep "$POLL_SECONDS"
  done
}

wait_for_selected_gpus
echo "[$(date '+%F %T')] starting QuickGELU 2GPU batch72 all-gather ImageNet run on GPUs ${GPUS}"
CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH=. python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  train_atas.py \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT"
