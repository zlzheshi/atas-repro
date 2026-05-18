#!/usr/bin/env bash
set -euo pipefail

RAW_DIR=${RAW_DIR:-/mnt/t1b6/xuzhejia/datasets/imagenet/raw}
TRAIN_URL=${TRAIN_URL:-}
DEVKIT_URL=${DEVKIT_URL:-}

mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

if [ -z "$TRAIN_URL" ]; then
  echo "TRAIN_URL is required."
  echo "Example:"
  echo "TRAIN_URL='<copied training images url>' DEVKIT_URL='<copied devkit url>' bash scripts/download_imagenet.sh"
  exit 1
fi

download_one() {
  local url=$1
  local output=$2
  echo "Downloading $output"
  aria2c \
    --continue=true \
    --max-connection-per-server=4 \
    --split=4 \
    --min-split-size=64M \
    --retry-wait=30 \
    --max-tries=0 \
    --timeout=60 \
    --connect-timeout=30 \
    --file-allocation=none \
    --allow-overwrite=false \
    --auto-file-renaming=false \
    --out="$output" \
    "$url"
}

download_one "$TRAIN_URL" ILSVRC2012_img_train.tar

if [ -n "$DEVKIT_URL" ]; then
  download_one "$DEVKIT_URL" ILSVRC2012_devkit_t12.tar.gz
fi

echo "Downloads saved under $RAW_DIR"

