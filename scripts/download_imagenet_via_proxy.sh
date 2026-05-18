#!/usr/bin/env bash
set -euo pipefail

RAW_DIR=${RAW_DIR:-/mnt/t1b6/xuzhejia/datasets/imagenet/raw}
PROXY=${PROXY:-127.0.0.1:18080}
TRAIN_URL=${TRAIN_URL:-https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar}
DEVKIT_URL=${DEVKIT_URL:-https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz}

EXPECTED_TRAIN_SIZE=147897477120
EXPECTED_TRAIN_MD5=1d675b47d978889d74fa0da5fadfb00e

mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

download_resume() {
  local url=$1
  local output=$2
  while true; do
    echo "[$(date '+%F %T')] downloading $output"
    curl \
      --socks5-hostname "$PROXY" \
      -L \
      -C - \
      --connect-timeout 30 \
      --retry 20 \
      --retry-delay 30 \
      --speed-time 180 \
      --speed-limit 1024 \
      -o "$output" \
      "$url" && break
    echo "[$(date '+%F %T')] curl failed for $output, retrying in 60s"
    sleep 60
  done
}

download_resume "$DEVKIT_URL" ILSVRC2012_devkit_t12.tar.gz
download_resume "$TRAIN_URL" ILSVRC2012_img_train.tar

actual_size=$(stat -c '%s' ILSVRC2012_img_train.tar)
echo "train size: $actual_size / $EXPECTED_TRAIN_SIZE"
if [ "$actual_size" -ne "$EXPECTED_TRAIN_SIZE" ]; then
  echo "Unexpected train tar size." >&2
  exit 2
fi

actual_md5=$(md5sum ILSVRC2012_img_train.tar | awk '{print $1}')
echo "train md5: $actual_md5"
if [ "$actual_md5" != "$EXPECTED_TRAIN_MD5" ]; then
  echo "Unexpected train tar md5." >&2
  exit 3
fi

echo "ImageNet download completed."
