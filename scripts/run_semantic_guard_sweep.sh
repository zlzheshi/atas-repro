#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/mnt/t1b6/xuzhejia/atas-repro}
CONDA_ROOT=${CONDA_ROOT:-/mnt/t1b6/xuzhejia/app/miniconda3}
DATA_ROOT=${DATA_ROOT:-/mnt/t1b6/xuzhejia/datasets/imagenet_subset_100x200/train}
VOC_ROOT=${VOC_ROOT:-/mnt/t1b6/xuzhejia/datasets/VOCdevkit/VOC2012}
FULL_CHECKPOINT_DIR=${FULL_CHECKPOINT_DIR:-outputs/atas_vitb_imagenet_full_author}
LOG_DIR=${LOG_DIR:-/mnt/t1b6/xuzhejia/logs}
MIN_FREE_MB=${MIN_FREE_MB:-30000}
POLL_SECONDS=${POLL_SECONDS:-120}

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate atas
cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"

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
    echo "[$(date '+%F %T')] waiting for a GPU with >= ${MIN_FREE_MB} MiB free memory..." >&2
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >&2
    sleep "$POLL_SECONDS"
  done
}

run_train() {
  local name=$1
  local config=$2
  local gpu
  gpu=$(wait_for_gpu)
  echo "[$(date '+%F %T')] train ${name} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. python train_atas.py \
    --config "$config" \
    --data-root "$DATA_ROOT" \
    2>&1 | tee "$LOG_DIR/${name}_train.log"
}

run_knn() {
  local name=$1
  local config=$2
  local checkpoint=$3
  local gpu
  gpu=$(wait_for_gpu)
  echo "[$(date '+%F %T')] kNN ${name} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. python scripts/evaluate_imagenet_subset_knn.py \
    --config "$config" \
    --data-root "$DATA_ROOT" \
    --checkpoint "$checkpoint" \
    --output-dir "outputs/eval_subset_100x200_${name}_knn" \
    --batch-size 256 \
    --num-workers 8 \
    2>&1 | tee "$LOG_DIR/${name}_knn.log"
}

run_voc() {
  local name=$1
  local config=$2
  local checkpoint=$3
  local dense_mode=$4
  local output_dir=$5
  local gpu
  gpu=$(wait_for_gpu)
  echo "[$(date '+%F %T')] VOC ${dense_mode} ${name} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. python scripts/evaluate_voc_zero_shot_seg.py \
    --config "$config" \
    --voc-root "$VOC_ROOT" \
    --checkpoint "$checkpoint" \
    --model-name "${name}_${dense_mode}" \
    --dense-mode "$dense_mode" \
    --output-dir "$output_dir" \
    2>&1 | tee "$LOG_DIR/${name}_voc_${dense_mode}.log"
}

run_drift() {
  local name=$1
  local config=$2
  local checkpoint=$3
  local gpu
  gpu=$(wait_for_gpu)
  echo "[$(date '+%F %T')] drift ${name} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. python scripts/diagnose_checkpoint_drift.py \
    --config "$config" \
    --data-root "$DATA_ROOT" \
    --checkpoint "full_epoch6=${FULL_CHECKPOINT_DIR}/checkpoint_epoch_6.pt" \
    --checkpoint "${name}=${checkpoint}" \
    --output-dir "outputs/checkpoint_drift_${name}" \
    --num-workers 0 \
    --max-batches 8 \
    --max-pairwise-patches 512 \
    2>&1 | tee "$LOG_DIR/${name}_drift.log"
}

run_one() {
  local name=$1
  local config=$2
  local checkpoint=$3
  run_train "$name" "$config"
  run_knn "$name" "$config" "$checkpoint"
  run_voc "$name" "$config" "$checkpoint" vanilla "outputs/voc_${name}"
  run_voc "$name" "$config" "$checkpoint" sclip "outputs/voc_sclip_${name}"
  run_drift "$name" "$config" "$checkpoint"
}

run_one \
  semantic_guard_gld010_ggd8_probe \
  configs/atas_vitb_subset_100x200_semantic_guard_gld010_ggd8_probe.yaml \
  outputs/atas_vitb_subset_100x200_semantic_guard_gld010_ggd8_probe/checkpoint_epoch_1.pt

run_one \
  semantic_guard_gld050_ggd4_probe \
  configs/atas_vitb_subset_100x200_semantic_guard_gld050_ggd4_probe.yaml \
  outputs/atas_vitb_subset_100x200_semantic_guard_gld050_ggd4_probe/checkpoint_epoch_1.pt

run_one \
  semantic_guard_320_probe \
  configs/atas_vitb_subset_100x200_semantic_guard_320_probe.yaml \
  outputs/atas_vitb_subset_100x200_semantic_guard_320_probe/checkpoint_epoch_1.pt

echo "[$(date '+%F %T')] semantic guard sweep finished"
