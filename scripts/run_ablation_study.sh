#!/usr/bin/env bash
set -euo pipefail

# ATAS ablation runner.
#
# 默认使用实验室服务器上的 ImageNet-100x200 子集，便于在较短时间内比较
# GLD / LLD / GGD / all-gather 的影响。若要做正式完整消融，可以把 DATA_ROOT
# 指向完整 ImageNet，并在 configs/ablation_*.yaml 中去掉 max_steps。

PROJECT_DIR=${PROJECT_DIR:-/mnt/t1b6/xuzhejia/atas-repro}
CONDA_ROOT=${CONDA_ROOT:-/mnt/t1b6/xuzhejia/app/miniconda3}
CONDA_ENV=${CONDA_ENV:-atas}
DATA_ROOT=${DATA_ROOT:-/mnt/t1b6/xuzhejia/datasets/imagenet_subset_100x200/train}
VOC_ROOT=${VOC_ROOT:-/mnt/t1b6/xuzhejia/datasets/VOCdevkit/VOC2012}
GPUS=${GPUS:-0,1}
NPROC=${NPROC:-$(awk -F, '{print NF}' <<< "$GPUS")}
VOC_LIMIT=${VOC_LIMIT:-}
DRIFT_BATCHES=${DRIFT_BATCHES:-4}
DRIFT_PATCHES=${DRIFT_PATCHES:-512}

if [[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

cd "$PROJECT_DIR"
mkdir -p outputs/ablation_study

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "DATA_ROOT not found: $DATA_ROOT" >&2
  exit 1
fi

if [[ ! -d "$VOC_ROOT" ]]; then
  echo "VOC_ROOT not found: $VOC_ROOT" >&2
  exit 1
fi

names=(
  full
  no_gld
  no_lld
  no_ggd
  no_allgather
)

configs=(
  configs/ablation_quickgelu_full.yaml
  configs/ablation_quickgelu_no_gld.yaml
  configs/ablation_quickgelu_no_lld.yaml
  configs/ablation_quickgelu_no_ggd.yaml
  configs/ablation_quickgelu_no_allgather.yaml
)

checkpoints=(
  outputs/ablation_quickgelu_full/checkpoint_epoch_1.pt
  outputs/ablation_quickgelu_no_gld/checkpoint_epoch_1.pt
  outputs/ablation_quickgelu_no_lld/checkpoint_epoch_1.pt
  outputs/ablation_quickgelu_no_ggd/checkpoint_epoch_1.pt
  outputs/ablation_quickgelu_no_allgather/checkpoint_epoch_1.pt
)

voc_limit_args=()
if [[ -n "$VOC_LIMIT" ]]; then
  voc_limit_args=(--limit "$VOC_LIMIT")
fi

run_train() {
  local name=$1
  local config=$2
  local checkpoint=$3

  if [[ -f "$checkpoint" ]]; then
    echo "[ablation] skip train ${name}; checkpoint exists: ${checkpoint}"
    return 0
  fi

  echo "[ablation] train ${name} with ${config}"
  CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH=. python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NPROC" \
    train_atas.py \
    --config "$config" \
    --data-root "$DATA_ROOT"
}

run_voc_eval() {
  local name=$1
  local config=$2
  local checkpoint=$3
  local dense_mode=$4
  local output_dir="outputs/ablation_study/voc_${name}_${dense_mode}"

  echo "[ablation] evaluate ${name} dense_mode=${dense_mode}"
  if [[ "$checkpoint" == "baseline" ]]; then
    CUDA_VISIBLE_DEVICES="${GPUS%%,*}" PYTHONPATH=. python scripts/evaluate_voc_zero_shot_seg.py \
      --config "$config" \
      --voc-root "$VOC_ROOT" \
      --model-name "ablation_${name}_${dense_mode}" \
      --dense-mode "$dense_mode" \
      --output-dir "$output_dir" \
      "${voc_limit_args[@]}"
  else
    CUDA_VISIBLE_DEVICES="${GPUS%%,*}" PYTHONPATH=. python scripts/evaluate_voc_zero_shot_seg.py \
      --config "$config" \
      --voc-root "$VOC_ROOT" \
      --checkpoint "$checkpoint" \
      --model-name "ablation_${name}_${dense_mode}" \
      --dense-mode "$dense_mode" \
      --output-dir "$output_dir" \
      "${voc_limit_args[@]}"
  fi
}

run_drift() {
  local name=$1
  local config=$2
  local checkpoint=$3

  echo "[ablation] drift diagnosis ${name}"
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" PYTHONPATH=. python scripts/diagnose_checkpoint_drift.py \
    --config "$config" \
    --data-root "$DATA_ROOT" \
    --checkpoint "${name}=${checkpoint}" \
    --output-dir "outputs/ablation_study/drift_${name}" \
    --num-workers 0 \
    --max-batches "$DRIFT_BATCHES" \
    --max-pairwise-patches "$DRIFT_PATCHES"
}

# CLIP baseline does not have a checkpoint. It provides the no-training reference
# for VOC dense prediction.
run_voc_eval baseline "${configs[0]}" baseline vanilla
run_voc_eval baseline "${configs[0]}" baseline sclip

for index in "${!names[@]}"; do
  name=${names[$index]}
  config=${configs[$index]}
  checkpoint=${checkpoints[$index]}
  run_train "$name" "$config" "$checkpoint"
  run_voc_eval "$name" "$config" "$checkpoint" vanilla
  run_voc_eval "$name" "$config" "$checkpoint" sclip
  run_drift "$name" "$config" "$checkpoint"
done

python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

root = Path("outputs/ablation_study")
names = ["baseline", "full", "no_gld", "no_lld", "no_ggd", "no_allgather"]

def read_metric(path: Path, key: str) -> str:
    if not path.exists():
        return "NA"
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    value = data.get(key)
    if isinstance(value, float):
        return f"{value:.4f}"
    return "NA"

lines = [
    "# ATAS Ablation Summary",
    "",
    "| Setting | Vanilla mIoU | SCLIP mIoU |",
    "| --- | ---: | ---: |",
]

for name in names:
    vanilla = read_metric(root / f"voc_{name}_vanilla" / "metrics.json", "foreground_miou")
    sclip = read_metric(root / f"voc_{name}_sclip" / "metrics.json", "foreground_miou")
    lines.append(f"| {name} | {vanilla} | {sclip} |")

lines.extend([
    "",
    "Drift diagnostics are stored under `outputs/ablation_study/drift_*`.",
])

(root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
