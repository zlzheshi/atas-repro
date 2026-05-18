#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/mnt/t1b6/xuzhejia/atas-repro}
CONDA_ROOT=${CONDA_ROOT:-/mnt/t1b6/xuzhejia/app/miniconda3}
VOC_ROOT=${VOC_ROOT:-/mnt/t1b6/xuzhejia/datasets/VOCdevkit/VOC2012}
CONFIG=${CONFIG:-configs/atas_vitb_imagenet_full_author.yaml}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-outputs/atas_vitb_imagenet_full_author}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/voc_sclip_full_author}
GPU=${GPU:-3}
IMAGE_SIZE=${IMAGE_SIZE:-448}
EPOCHS=${EPOCHS:-1 3 6}
LIMIT=${LIMIT:-}

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate atas
cd "$PROJECT_DIR"

limit_args=()
if [[ -n "$LIMIT" ]]; then
  limit_args=(--limit "$LIMIT")
fi

mkdir -p "$OUTPUT_ROOT"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. python scripts/evaluate_voc_zero_shot_seg.py \
  --config "$CONFIG" \
  --voc-root "$VOC_ROOT" \
  --model-name openclip_baseline_sclip \
  --output-dir "$OUTPUT_ROOT/baseline" \
  --image-size "$IMAGE_SIZE" \
  --dense-mode sclip \
  "${limit_args[@]}"

for epoch in $EPOCHS; do
  checkpoint="$CHECKPOINT_DIR/checkpoint_epoch_${epoch}.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "skip missing checkpoint: $checkpoint"
    continue
  fi

  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. python scripts/evaluate_voc_zero_shot_seg.py \
    --config "$CONFIG" \
    --voc-root "$VOC_ROOT" \
    --checkpoint "$checkpoint" \
    --model-name "atas_full_imagenet_epoch${epoch}_sclip" \
    --output-dir "$OUTPUT_ROOT/epoch_${epoch}" \
    --image-size "$IMAGE_SIZE" \
    --dense-mode sclip \
    "${limit_args[@]}"
done

PYTHONPATH=. python scripts/summarize_voc_sweep.py --root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/summary.csv"
