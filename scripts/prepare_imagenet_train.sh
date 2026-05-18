#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/t1b6/xuzhejia/datasets/imagenet}
RAW_DIR="$ROOT/raw"
TRAIN_TAR="$RAW_DIR/ILSVRC2012_img_train.tar"
TRAIN_TAR_DIR="$ROOT/train_tars"
TRAIN_DIR="$ROOT/train"
EXPECTED_MD5=${EXPECTED_MD5:-1d675b47d978889d74fa0da5fadfb00e}
SKIP_MD5=${SKIP_MD5:-0}

mkdir -p "$TRAIN_TAR_DIR" "$TRAIN_DIR"

if [ ! -f "$TRAIN_TAR" ]; then
  echo "Missing $TRAIN_TAR" >&2
  exit 1
fi

if [ "$SKIP_MD5" = "1" ]; then
  echo "Skipping md5 check because SKIP_MD5=1"
else
  echo "Checking md5 for $TRAIN_TAR"
  actual_md5=$(md5sum "$TRAIN_TAR" | awk '{print $1}')
  echo "md5: $actual_md5"
  if [ "$actual_md5" != "$EXPECTED_MD5" ]; then
    echo "Unexpected md5. Expected $EXPECTED_MD5" >&2
    exit 2
  fi
fi

if [ "$(find "$TRAIN_TAR_DIR" -maxdepth 1 -name 'n*.tar' | wc -l)" -lt 1000 ]; then
  echo "Extracting outer train tar into $TRAIN_TAR_DIR"
  tar -xf "$TRAIN_TAR" -C "$TRAIN_TAR_DIR"
fi

echo "Extracting class tar files into $TRAIN_DIR"
count=0
for class_tar in "$TRAIN_TAR_DIR"/n*.tar; do
  class_id=$(basename "$class_tar" .tar)
  class_dir="$TRAIN_DIR/$class_id"
  done_file="$class_dir/.extract_done"

  if [ -f "$done_file" ]; then
    continue
  fi

  mkdir -p "$class_dir"
  tar -xf "$class_tar" -C "$class_dir"
  touch "$done_file"
  count=$((count + 1))

  if [ $((count % 25)) -eq 0 ]; then
    echo "Extracted $count class archives in this run"
  fi
done

class_count=$(find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
image_count=$(find "$TRAIN_DIR" -type f -name '*.JPEG' | wc -l)

echo "ImageNet train prepared."
echo "classes: $class_count"
echo "images:  $image_count"
echo "path:    $TRAIN_DIR"
