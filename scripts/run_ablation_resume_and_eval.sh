#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/mnt/t1b6/xuzhejia/atas-repro}
CONDA_ROOT=${CONDA_ROOT:-/mnt/t1b6/xuzhejia/app/miniconda3}
DATA_ROOT=${DATA_ROOT:-/mnt/t1b6/xuzhejia/datasets/imagenet_subset_100x200/train}
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

run_resume() {
  local name=$1
  local config=$2
  local ckpt=$3
  local log_file="$LOG_DIR/atas_subset_100x200_${name}_resume_epoch3.log"
  local gpu
  gpu=$(wait_for_gpu)
  echo "[$(date '+%F %T')] running ${name} resume on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" python train_atas.py \
    --config "$config" \
    --data-root "$DATA_ROOT" \
    --resume "$ckpt" \
    2>&1 | tee "$log_file"
}

run_eval() {
  local name=$1
  local config=$2
  local ckpt=$3
  local output_dir=$4
  local log_file="$LOG_DIR/eval_subset_100x200_${name}_knn.log"
  local gpu
  gpu=$(wait_for_gpu)
  echo "[$(date '+%F %T')] evaluating ${name} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. python scripts/evaluate_imagenet_subset_knn.py \
    --config "$config" \
    --data-root "$DATA_ROOT" \
    --checkpoint "$ckpt" \
    --output-dir "$output_dir" \
    --batch-size 256 \
    --num-workers 8 \
    2>&1 | tee "$log_file"
}

run_resume \
  gld_only \
  configs/atas_vitb_subset_100x200_gld_only.yaml \
  outputs/atas_vitb_subset_100x200_gld_only/checkpoint_epoch_2.pt

run_eval \
  gld_only \
  configs/atas_vitb_subset_100x200_gld_only.yaml \
  outputs/atas_vitb_subset_100x200_gld_only/checkpoint_epoch_3.pt \
  outputs/eval_subset_100x200_gld_only_knn

run_resume \
  gld_lld \
  configs/atas_vitb_subset_100x200_gld_lld.yaml \
  outputs/atas_vitb_subset_100x200_gld_lld/checkpoint_epoch_2.pt

run_eval \
  gld_lld \
  configs/atas_vitb_subset_100x200_gld_lld.yaml \
  outputs/atas_vitb_subset_100x200_gld_lld/checkpoint_epoch_3.pt \
  outputs/eval_subset_100x200_gld_lld_knn

python scripts/summarize_knn_metrics.py \
  --runs \
  "Full ATAS=outputs/eval_subset_100x200_knn/metrics.json" \
  "GLD only=outputs/eval_subset_100x200_gld_only_knn/metrics.json" \
  "GLD + LLD=outputs/eval_subset_100x200_gld_lld_knn/metrics.json" \
  --output-dir outputs/ablation_summary

echo "[$(date '+%F %T')] ablation resume and eval finished"
