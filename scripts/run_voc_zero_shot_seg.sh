#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/mnt/t1b6/xuzhejia/atas-repro}
CONDA_ROOT=${CONDA_ROOT:-/mnt/t1b6/xuzhejia/app/miniconda3}
VOC_ROOT=${VOC_ROOT:-/mnt/t1b6/xuzhejia/datasets/VOCdevkit/VOC2012}
GPU=${GPU:-2}
IMAGE_SIZE=${IMAGE_SIZE:-448}
LIMIT=${LIMIT:-}

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate atas
cd "$PROJECT_DIR"

limit_args=()
if [[ -n "$LIMIT" ]]; then
  limit_args=(--limit "$LIMIT")
fi

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. python scripts/evaluate_voc_zero_shot_seg.py \
  --config configs/atas_vitb_subset_100x200_stable.yaml \
  --voc-root "$VOC_ROOT" \
  --model-name openclip_baseline \
  --output-dir outputs/voc_zero_shot_seg_baseline \
  --image-size "$IMAGE_SIZE" \
  "${limit_args[@]}"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. python scripts/evaluate_voc_zero_shot_seg.py \
  --config configs/atas_vitb_subset_100x200_stable.yaml \
  --voc-root "$VOC_ROOT" \
  --checkpoint outputs/atas_vitb_subset_100x200_stable/checkpoint_epoch_3.pt \
  --model-name full_atas_epoch3 \
  --output-dir outputs/voc_zero_shot_seg_full_atas \
  --image-size "$IMAGE_SIZE" \
  "${limit_args[@]}"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. python scripts/evaluate_voc_zero_shot_seg.py \
  --config configs/atas_vitb_subset_100x200_stable.yaml \
  --voc-root "$VOC_ROOT" \
  --checkpoint outputs/atas_vitb_subset_100x200_stable/checkpoint_epoch_6.pt \
  --model-name full_atas_epoch6 \
  --output-dir outputs/voc_zero_shot_seg_full_atas_epoch6 \
  --image-size "$IMAGE_SIZE" \
  "${limit_args[@]}"
