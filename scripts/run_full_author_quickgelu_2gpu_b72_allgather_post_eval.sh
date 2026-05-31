#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/mnt/t1b6/xuzhejia/atas-repro}
CONDA_ROOT=${CONDA_ROOT:-/mnt/t1b6/xuzhejia/app/miniconda3}
DATA_ROOT=${DATA_ROOT:-/mnt/t1b6/xuzhejia/datasets/imagenet/train}
VOC_ROOT=${VOC_ROOT:-/mnt/t1b6/xuzhejia/datasets/VOCdevkit/VOC2012}
LOG_DIR=${LOG_DIR:-/mnt/t1b6/xuzhejia/logs}
CONFIG=${CONFIG:-configs/atas_vitb_imagenet_full_author_quickgelu_2gpu_b72_allgather.yaml}
CHECKPOINT=${CHECKPOINT:-outputs/atas_vitb_imagenet_full_author_quickgelu_2gpu_b72_allgather/checkpoint_epoch_6.pt}
GPU=${GPU:-0}
POLL_SECONDS=${POLL_SECONDS:-120}

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate atas
cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"

while [[ ! -f "$CHECKPOINT" ]]; do
  echo "[$(date '+%F %T')] waiting for ${CHECKPOINT}..."
  sleep "$POLL_SECONDS"
done

echo "[$(date '+%F %T')] evaluating QuickGELU b72 all-gather epoch6 vanilla on GPU ${GPU}"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. python scripts/evaluate_voc_zero_shot_seg.py \
  --config "$CONFIG" \
  --voc-root "$VOC_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --model-name atas_quickgelu_b72_allgather_epoch6 \
  --dense-mode vanilla \
  --output-dir outputs/voc_full_author_quickgelu_b72_allgather_epoch6

echo "[$(date '+%F %T')] evaluating QuickGELU b72 all-gather epoch6 SCLIP on GPU ${GPU}"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. python scripts/evaluate_voc_zero_shot_seg.py \
  --config "$CONFIG" \
  --voc-root "$VOC_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --model-name atas_quickgelu_b72_allgather_epoch6_sclip \
  --dense-mode sclip \
  --output-dir outputs/voc_sclip_full_author_quickgelu_b72_allgather_epoch6

echo "[$(date '+%F %T')] diagnosing QuickGELU b72 all-gather epoch6 drift on GPU ${GPU}"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. python scripts/diagnose_checkpoint_drift.py \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --checkpoint quickgelu_b72_allgather_epoch6="$CHECKPOINT" \
  --output-dir outputs/checkpoint_drift_full_author_quickgelu_b72_allgather_epoch6 \
  --num-workers 0 \
  --max-batches 8 \
  --max-pairwise-patches 512

echo "[$(date '+%F %T')] QuickGELU b72 all-gather post evaluation finished"
