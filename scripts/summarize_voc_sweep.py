from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        with open(metrics_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)
        rows.append(
            {
                "run": metrics_path.parent.name,
                "model": metrics.get("model", metrics_path.parent.name),
                "foreground_miou": metrics.get("foreground_miou", ""),
                "foreground_pixel_acc": metrics.get("foreground_pixel_acc", ""),
                "mean_class_acc": metrics.get("mean_class_acc", ""),
            }
        )

    rows.sort(key=lambda item: float(item["foreground_miou"]), reverse=True)
    fieldnames = ["run", "model", "foreground_miou", "foreground_pixel_acc", "mean_class_acc"]

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(",".join(fieldnames))
    for row in rows:
        print(",".join(str(row[name]) for name in fieldnames))


if __name__ == "__main__":
    main()
