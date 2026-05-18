from __future__ import annotations

import argparse
import os
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an ImageFolder-style ImageNet subset using symlinks.")
    parser.add_argument("--source", default="/mnt/t1b6/xuzhejia/datasets/imagenet/train")
    parser.add_argument("--output", default="/mnt/t1b6/xuzhejia/datasets/imagenet_subset_100x200/train")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--images-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def link_file(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    os.symlink(source, destination)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source)
    output_root = Path(args.output)

    if not source_root.exists():
        raise FileNotFoundError(f"source not found: {source_root}")

    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output_root}. Use --overwrite to rebuild.")

    output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    classes = sorted(path for path in source_root.iterdir() if path.is_dir())
    if len(classes) < args.num_classes:
        raise ValueError(f"requested {args.num_classes} classes, only found {len(classes)}")

    selected_classes = rng.sample(classes, args.num_classes)
    selected_classes = sorted(selected_classes, key=lambda path: path.name)

    total_images = 0
    for class_dir in selected_classes:
        images = sorted(class_dir.glob("*.JPEG"))
        if len(images) > args.images_per_class:
            images = rng.sample(images, args.images_per_class)
            images = sorted(images, key=lambda path: path.name)

        target_class_dir = output_root / class_dir.name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        for image in images:
            link_file(image, target_class_dir / image.name)

        total_images += len(images)

    manifest = output_root.parent / "manifest.txt"
    with manifest.open("w", encoding="utf-8") as file:
        file.write(f"source={source_root}\n")
        file.write(f"output={output_root}\n")
        file.write(f"num_classes={len(selected_classes)}\n")
        file.write(f"images={total_images}\n")
        file.write(f"images_per_class_limit={args.images_per_class}\n")
        file.write(f"seed={args.seed}\n")
        file.write("classes=\n")
        for class_dir in selected_classes:
            file.write(f"{class_dir.name}\n")

    print(f"subset path: {output_root}")
    print(f"classes: {len(selected_classes)}")
    print(f"images: {total_images}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()

