#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/mnt/t1b6/xuzhejia/atas-repro}
CONDA_ROOT=${CONDA_ROOT:-/mnt/t1b6/xuzhejia/app/miniconda3}
DATA_ROOT=${DATA_ROOT:-/mnt/t1b6/xuzhejia/datasets/imagenet_subset_100x200/train}
VOC_ROOT=${VOC_ROOT:-/mnt/t1b6/xuzhejia/datasets/VOCdevkit/VOC2012}
LOG_DIR=${LOG_DIR:-/mnt/t1b6/xuzhejia/logs}
NAME=${NAME:-quickgelu_allgather_probe}
CONFIG=${CONFIG:-configs/atas_vitb_subset_100x200_quickgelu_allgather_probe.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/atas_vitb_subset_100x200_quickgelu_allgather_probe}
CHECKPOINT=${CHECKPOINT:-${OUTPUT_DIR}/checkpoint_epoch_1.pt}
GPUS=${GPUS:-0,1}

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate atas
cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"

echo "[$(date '+%F %T')] train QuickGELU all-gather probe on GPUs ${GPUS}"
CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH=. python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  train_atas.py \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  2>&1 | tee "$LOG_DIR/${NAME}_train.log"

echo "[$(date '+%F %T')] evaluate QuickGELU all-gather probe kNN"
CUDA_VISIBLE_DEVICES="${GPUS%%,*}" PYTHONPATH=. python scripts/evaluate_imagenet_subset_knn.py \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "outputs/eval_subset_100x200_${NAME}_knn" \
  --batch-size 256 \
  --num-workers 8 \
  2>&1 | tee "$LOG_DIR/${NAME}_knn.log"

echo "[$(date '+%F %T')] evaluate QuickGELU all-gather probe VOC vanilla"
CUDA_VISIBLE_DEVICES="${GPUS%%,*}" PYTHONPATH=. python scripts/evaluate_voc_zero_shot_seg.py \
  --config "$CONFIG" \
  --voc-root "$VOC_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --model-name "${NAME}_vanilla" \
  --dense-mode vanilla \
  --output-dir "outputs/voc_${NAME}" \
  2>&1 | tee "$LOG_DIR/${NAME}_voc_vanilla.log"

echo "[$(date '+%F %T')] evaluate QuickGELU all-gather probe VOC SCLIP"
CUDA_VISIBLE_DEVICES="${GPUS%%,*}" PYTHONPATH=. python scripts/evaluate_voc_zero_shot_seg.py \
  --config "$CONFIG" \
  --voc-root "$VOC_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --model-name "${NAME}_sclip" \
  --dense-mode sclip \
  --output-dir "outputs/voc_sclip_${NAME}" \
  2>&1 | tee "$LOG_DIR/${NAME}_voc_sclip.log"

echo "[$(date '+%F %T')] diagnose QuickGELU all-gather probe drift"
CUDA_VISIBLE_DEVICES="${GPUS%%,*}" PYTHONPATH=. python scripts/diagnose_checkpoint_drift.py \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --checkpoint "${NAME}=${CHECKPOINT}" \
  --output-dir "outputs/checkpoint_drift_${NAME}" \
  --num-workers 0 \
  --max-batches 8 \
  --max-pairwise-patches 512 \
  2>&1 | tee "$LOG_DIR/${NAME}_drift.log"

echo "[$(date '+%F %T')] QuickGELU all-gather probe finished"
