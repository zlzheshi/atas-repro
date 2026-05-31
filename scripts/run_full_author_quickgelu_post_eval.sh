#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/mnt/t1b6/xuzhejia/atas-repro}
CONDA_ROOT=${CONDA_ROOT:-/mnt/t1b6/xuzhejia/app/miniconda3}
DATA_ROOT=${DATA_ROOT:-/mnt/t1b6/xuzhejia/datasets/imagenet_subset_100x200/train}
VOC_ROOT=${VOC_ROOT:-/mnt/t1b6/xuzhejia/datasets/VOCdevkit/VOC2012}
LOG_DIR=${LOG_DIR:-/mnt/t1b6/xuzhejia/logs}
CHECKPOINT=${CHECKPOINT:-outputs/atas_vitb_imagenet_full_author_quickgelu/checkpoint_epoch_6.pt}
MIN_FREE_MB=${MIN_FREE_MB:-30000}
POLL_SECONDS=${POLL_SECONDS:-120}

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate atas
cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"

wait_for_file() {
  local file=$1
  while [[ ! -f "$file" ]]; do
    echo "[$(date '+%F %T')] waiting for ${file}..."
    sleep "$POLL_SECONDS"
  done
}

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -v min_free="$MIN_FREE_MB" -F ', ' '$2 >= min_free {print $1; exit}'
}

wait_for_gpu() {
  local gpu
  while true; do
    gpu=$(pick_gpu || true)
    if [[ -n "${gpu:-}" ]]; then
      echo "$gpu"
      return 0
    fi
    echo "[$(date '+%F %T')] waiting for one GPU with >= ${MIN_FREE_MB} MiB free memory..."
    sleep "$POLL_SECONDS"
  done
}

wait_for_file "$CHECKPOINT"

GPU=$(wait_for_gpu)
echo "[$(date '+%F %T')] evaluating QuickGELU epoch6 vanilla on GPU ${GPU}"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. python scripts/evaluate_voc_zero_shot_seg.py \
  --config configs/atas_vitb_imagenet_full_author_quickgelu.yaml \
  --voc-root "$VOC_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --model-name atas_quickgelu_epoch6 \
  --output-dir outputs/voc_full_author_quickgelu_epoch6 \
  2>&1 | tee "$LOG_DIR/voc_full_author_quickgelu_epoch6.log"

GPU=$(wait_for_gpu)
echo "[$(date '+%F %T')] evaluating QuickGELU epoch6 SCLIP on GPU ${GPU}"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. python scripts/evaluate_voc_zero_shot_seg.py \
  --config configs/atas_vitb_imagenet_full_author_quickgelu.yaml \
  --voc-root "$VOC_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --model-name atas_quickgelu_epoch6_sclip \
  --dense-mode sclip \
  --output-dir outputs/voc_sclip_full_author_quickgelu_epoch6 \
  2>&1 | tee "$LOG_DIR/voc_sclip_full_author_quickgelu_epoch6.log"

GPU=$(wait_for_gpu)
echo "[$(date '+%F %T')] diagnosing QuickGELU epoch6 drift on GPU ${GPU}"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. python scripts/diagnose_checkpoint_drift.py \
  --config configs/atas_vitb_imagenet_full_author_quickgelu.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint "quickgelu_epoch6=${CHECKPOINT}" \
  --output-dir outputs/checkpoint_drift_full_author_quickgelu_epoch6 \
  --num-workers 0 \
  --max-batches 8 \
  --max-pairwise-patches 512 \
  2>&1 | tee "$LOG_DIR/checkpoint_drift_full_author_quickgelu_epoch6.log"

echo "[$(date '+%F %T')] QuickGELU post evaluation finished"
