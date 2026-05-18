#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/mnt/t1b6/xuzhejia/datasets}
VOC_DIR="$DATA_ROOT/VOCdevkit/VOC2012"
ARCHIVE="$DATA_ROOT/VOCtrainval_11-May-2012.tar"
URL=${URL:-http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar}

mkdir -p "$DATA_ROOT"

if [[ -d "$VOC_DIR/JPEGImages" && -d "$VOC_DIR/SegmentationClass" ]]; then
  echo "VOC2012 already exists: $VOC_DIR"
  exit 0
fi

echo "Downloading/resuming VOC2012 to $ARCHIVE"
wget -c "$URL" -O "$ARCHIVE"

echo "Extracting $ARCHIVE"
tar -xf "$ARCHIVE" -C "$DATA_ROOT"
echo "VOC2012 ready: $VOC_DIR"
