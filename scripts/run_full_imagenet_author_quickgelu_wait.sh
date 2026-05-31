#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/mnt/t1b6/xuzhejia/atas-repro}
CONDA_ROOT=${CONDA_ROOT:-/mnt/t1b6/xuzhejia/app/miniconda3}
DATA_ROOT=${DATA_ROOT:-/mnt/t1b6/xuzhejia/datasets/imagenet/train}
LOG_DIR=${LOG_DIR:-/mnt/t1b6/xuzhejia/logs}
MIN_FREE_MB=${MIN_FREE_MB:-30000}
POLL_SECONDS=${POLL_SECONDS:-120}

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate atas
cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"

wait_for_four_gpus() {
  while true; do
    mapfile -t gpus < <(
      nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -v min_free="$MIN_FREE_MB" -F ', ' '$2 >= min_free {print $1}'
    )
    if (( ${#gpus[@]} >= 4 )); then
      IFS=,
      echo "${gpus[*]:0:4}"
      return 0
    fi
    echo "[$(date '+%F %T')] waiting for 4 GPUs with >= ${MIN_FREE_MB} MiB free memory..." >&2
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >&2
    sleep "$POLL_SECONDS"
  done
}

GPU_LIST=$(wait_for_four_gpus)
echo "[$(date '+%F %T')] starting author-aligned QuickGELU ImageNet run on GPUs ${GPU_LIST}"
CUDA_VISIBLE_DEVICES="$GPU_LIST" PYTHONPATH=. \
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  train_atas.py \
  --config configs/atas_vitb_imagenet_full_author_quickgelu.yaml \
  --data-root "$DATA_ROOT"
