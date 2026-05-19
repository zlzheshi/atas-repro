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

run_drift_diagnostic() {
  local output_dir=$1
  local log_file=$2
  local gpu
  gpu=$(wait_for_gpu)
  echo "[$(date '+%F %T')] running drift diagnostic on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. python scripts/diagnose_checkpoint_drift.py \
    --config configs/atas_vitb_subset_100x200_stable.yaml \
    --data-root "$DATA_ROOT" \
    --checkpoint "full_epoch1=${FULL_CHECKPOINT_DIR}/checkpoint_epoch_1.pt" \
    --checkpoint "full_epoch6=${FULL_CHECKPOINT_DIR}/checkpoint_epoch_6.pt" \
    --output-dir "$output_dir" \
    --max-batches 8 \
    --max-pairwise-patches 512 \
    2>&1 | tee "$log_file"
}

run_drift_diagnostic_with_probe() {
  local output_dir=$1
  local log_file=$2
  local gpu
  gpu=$(wait_for_gpu)
  echo "[$(date '+%F %T')] running drift diagnostic with semantic guard probe on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. python scripts/diagnose_checkpoint_drift.py \
    --config configs/atas_vitb_subset_100x200_stable.yaml \
    --data-root "$DATA_ROOT" \
    --checkpoint "full_epoch1=${FULL_CHECKPOINT_DIR}/checkpoint_epoch_1.pt" \
    --checkpoint "full_epoch6=${FULL_CHECKPOINT_DIR}/checkpoint_epoch_6.pt" \
    --checkpoint "semantic_guard_probe=outputs/atas_vitb_subset_100x200_semantic_guard_probe/checkpoint_epoch_1.pt" \
    --output-dir "$output_dir" \
    --max-batches 8 \
    --max-pairwise-patches 512 \
    2>&1 | tee "$log_file"
}

run_semantic_guard_probe() {
  local gpu
  gpu=$(wait_for_gpu)
  echo "[$(date '+%F %T')] running semantic guard subset probe on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. python train_atas.py \
    --config configs/atas_vitb_subset_100x200_semantic_guard_probe.yaml \
    --data-root "$DATA_ROOT" \
    2>&1 | tee "$LOG_DIR/atas_semantic_guard_probe_train.log"
}

run_knn_eval() {
  local gpu
  gpu=$(wait_for_gpu)
  echo "[$(date '+%F %T')] evaluating semantic guard probe kNN on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. python scripts/evaluate_imagenet_subset_knn.py \
    --config configs/atas_vitb_subset_100x200_semantic_guard_probe.yaml \
    --data-root "$DATA_ROOT" \
    --checkpoint outputs/atas_vitb_subset_100x200_semantic_guard_probe/checkpoint_epoch_1.pt \
    --output-dir outputs/eval_subset_100x200_semantic_guard_probe_knn \
    --batch-size 256 \
    --num-workers 8 \
    2>&1 | tee "$LOG_DIR/eval_semantic_guard_probe_knn.log"
}

run_voc_eval() {
  local gpu
  gpu=$(wait_for_gpu)
  echo "[$(date '+%F %T')] evaluating semantic guard probe on VOC2012 with vanilla patch matching on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. python scripts/evaluate_voc_zero_shot_seg.py \
    --config configs/atas_vitb_subset_100x200_semantic_guard_probe.yaml \
    --voc-root "$VOC_ROOT" \
    --checkpoint outputs/atas_vitb_subset_100x200_semantic_guard_probe/checkpoint_epoch_1.pt \
    --model-name semantic_guard_probe \
    --output-dir outputs/voc_semantic_guard_probe \
    2>&1 | tee "$LOG_DIR/eval_semantic_guard_probe_voc.log"
}

run_drift_diagnostic \
  outputs/checkpoint_drift_full_author_subset \
  "$LOG_DIR/checkpoint_drift_full_author_subset.log"

run_semantic_guard_probe
run_knn_eval
run_voc_eval

run_drift_diagnostic_with_probe \
  outputs/checkpoint_drift_full_author_subset_after_probe \
  "$LOG_DIR/checkpoint_drift_full_author_subset_after_probe.log"

echo "[$(date '+%F %T')] semantic guard probe workflow finished"
