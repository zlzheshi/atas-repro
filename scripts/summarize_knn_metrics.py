from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="Items like name=path/to/metrics.json")
    parser.add_argument("--output-dir", default="outputs/ablation_summary")
    return parser.parse_args()


def load_main_row(path: Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as file:
        rows = json.load(file)
    for row in rows:
        model = str(row.get("model", ""))
        if model != "openclip_baseline" and "drift" not in model:
            return row
    raise ValueError(f"no checkpoint row found in {path}")


def load_baseline_row(path: Path) -> dict[str, object] | None:
    with open(path, "r", encoding="utf-8") as file:
        rows = json.load(file)
    for row in rows:
        if row.get("model") == "openclip_baseline":
            return row
    return None


def format_float(value: object) -> str:
    if value == "":
        return ""
    return f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed_runs: list[tuple[str, Path]] = []
    for item in args.runs:
        if "=" not in item:
            raise ValueError(f"run must be name=path, got {item}")
        name, path = item.split("=", 1)
        parsed_runs.append((name, Path(path)))

    rows: list[dict[str, object]] = []
    baseline = load_baseline_row(parsed_runs[0][1])
    if baseline is not None:
        rows.append({"run": "OpenCLIP baseline", **baseline})

    for name, path in parsed_runs:
        rows.append({"run": name, **load_main_row(path)})

    fields = [
        "run",
        "model",
        "top1_knn",
        "top5_neighbor_hit",
        "mean_top1_similarity",
        "mean_own_centroid_similarity",
        "mean_nearest_other_centroid_similarity",
        "centroid_margin",
    ]
    with open(output_dir / "ablation_metrics.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    with open(output_dir / "ablation_metrics.md", "w", encoding="utf-8") as file:
        file.write("# ATAS Ablation kNN Summary\n\n")
        file.write("| Run | Top-1 kNN | Top-5 Hit | Centroid Margin |\n")
        file.write("| --- | ---: | ---: | ---: |\n")
        for row in rows:
            file.write(
                f"| {row['run']} | "
                f"{format_float(row.get('top1_knn', ''))} | "
                f"{format_float(row.get('top5_neighbor_hit', ''))} | "
                f"{format_float(row.get('centroid_margin', ''))} |\n"
            )

    with open(output_dir / "ablation_metrics.json", "w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)

    print(f"saved {output_dir.resolve()}")


if __name__ == "__main__":
    main()
