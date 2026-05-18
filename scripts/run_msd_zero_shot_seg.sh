#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/mnt/t1b6/xuzhejia/atas-repro}
CONDA_ROOT=${CONDA_ROOT:-/mnt/t1b6/xuzhejia/app/miniconda3}
DATA_ROOT=${DATA_ROOT:-/data/haier/wangcairui/dataset/MSD}
GPU=${GPU:-2}
IMAGE_SIZE=${IMAGE_SIZE:-448}
SPLIT=${SPLIT:-val}
LIMIT=${LIMIT:-}
SAVE_VIS=${SAVE_VIS:-0}
SAVE_VIS_PER_CLASS=${SAVE_VIS_PER_CLASS:-4}
BG_CLASS=${BG_CLASS:-}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-}

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate atas
cd "$PROJECT_DIR"

limit_args=()
if [[ -n "$LIMIT" ]]; then
  limit_args=(--limit "$LIMIT")
fi

background_args=()
if [[ -n "$BG_CLASS" ]]; then
  background_args=(--background-class "$BG_CLASS")
  if [[ -z "$OUTPUT_SUFFIX" ]]; then
    OUTPUT_SUFFIX="_with_bg"
  fi
fi

common_args=(
  --config configs/atas_vitb_subset_100x200_stable.yaml
  --data-root "$DATA_ROOT"
  --classes "oil stain" "stain" "scratch"
  "${background_args[@]}"
  --split "$SPLIT"
  --image-size "$IMAGE_SIZE"
  --save-vis "$SAVE_VIS"
  --save-vis-per-class "$SAVE_VIS_PER_CLASS"
  "${limit_args[@]}"
)

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. python scripts/evaluate_voc_like_zero_shot_seg.py \
  "${common_args[@]}" \
  --model-name openclip_baseline \
  --output-dir "outputs/msd_zero_shot_seg_baseline${OUTPUT_SUFFIX}"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. python scripts/evaluate_voc_like_zero_shot_seg.py \
  "${common_args[@]}" \
  --checkpoint outputs/atas_vitb_subset_100x200_stable/checkpoint_epoch_3.pt \
  --model-name full_atas_epoch3 \
  --output-dir "outputs/msd_zero_shot_seg_full_atas${OUTPUT_SUFFIX}"
