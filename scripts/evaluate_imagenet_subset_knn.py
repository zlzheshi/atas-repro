from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/eval_subset_knn")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gallery-per-class", type=int, default=160)
    parser.add_argument("--query-per-class", type=int, default=40)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_split(
    dataset: ImageFolder,
    gallery_per_class: int,
    query_per_class: int,
    max_classes: int | None,
    seed: int,
) -> tuple[list[int], list[int], dict[str, int]]:
    class_to_indices: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(dataset.targets):
        class_to_indices[int(label)].append(index)

    labels = sorted(class_to_indices)
    if max_classes is not None:
        labels = labels[:max_classes]

    rng = random.Random(seed)
    gallery_indices: list[int] = []
    query_indices: list[int] = []
    per_class_counts: dict[str, int] = {}

    for label in labels:
        indices = list(class_to_indices[label])
        rng.shuffle(indices)
        needed = gallery_per_class + query_per_class
        if len(indices) < needed:
            raise ValueError(
                f"class {dataset.classes[label]} only has {len(indices)} images, "
                f"but {needed} are required"
            )

        selected = indices[:needed]
        gallery_indices.extend(selected[:gallery_per_class])
        query_indices.extend(selected[gallery_per_class:])
        per_class_counts[dataset.classes[label]] = len(selected)

    return gallery_indices, query_indices, per_class_counts


def load_model(config: dict, checkpoint: str | None, device: torch.device) -> tuple[nn.Module, object]:
    model_name = config["model"]["name"]
    pretrained = config["model"]["pretrained"]
    force_quick_gelu = bool(config["model"].get("quick_gelu", False))
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        force_quick_gelu=force_quick_gelu,
    )

    if checkpoint is not None:
        loaded = torch.load(checkpoint, map_location="cpu")
        visual_state = loaded.get("visual_state_dict", loaded.get("model", loaded))
        missing, unexpected = model.visual.load_state_dict(visual_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint load mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
            )

    model = model.to(device).eval()
    return model, preprocess_val


@torch.no_grad()
def extract_features(
    model: nn.Module,
    dataset: ImageFolder,
    indices: list[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for images, target in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features.float(), dim=-1)
        features.append(image_features.cpu())
        labels.append(target.cpu())

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def knn_metrics(
    gallery_features: torch.Tensor,
    gallery_labels: torch.Tensor,
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    device: torch.device,
    chunk_size: int = 512,
) -> dict[str, float]:
    gallery_features = gallery_features.to(device)
    gallery_labels = gallery_labels.to(device)
    query_features = query_features.to(device)
    query_labels = query_labels.to(device)

    total = 0
    top1_correct = 0
    top5_hit = 0
    max_sim_sum = 0.0

    for start in range(0, query_features.shape[0], chunk_size):
        end = min(start + chunk_size, query_features.shape[0])
        sims = query_features[start:end] @ gallery_features.t()
        top5 = sims.topk(k=5, dim=1)
        neighbor_labels = gallery_labels[top5.indices]
        targets = query_labels[start:end, None]

        top1_correct += (neighbor_labels[:, 0:1] == targets).sum().item()
        top5_hit += (neighbor_labels == targets).any(dim=1).sum().item()
        max_sim_sum += top5.values[:, 0].sum().item()
        total += end - start

    return {
        "top1_knn": top1_correct / total,
        "top5_neighbor_hit": top5_hit / total,
        "mean_top1_similarity": max_sim_sum / total,
    }


def centroid_metrics(features: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    labels = labels.long()
    class_ids = labels.unique(sorted=True)
    centroids = []
    for class_id in class_ids:
        centroid = features[labels == class_id].mean(dim=0)
        centroids.append(F.normalize(centroid, dim=0))
    centroids_tensor = torch.stack(centroids)
    class_to_row = {int(class_id): row for row, class_id in enumerate(class_ids.tolist())}

    own_sims = []
    nearest_other_sims = []
    for feature, label in zip(features, labels):
        sims = feature @ centroids_tensor.t()
        own_row = class_to_row[int(label)]
        own_sims.append(float(sims[own_row]))
        sims[own_row] = -1.0
        nearest_other_sims.append(float(sims.max()))

    return {
        "mean_own_centroid_similarity": sum(own_sims) / len(own_sims),
        "mean_nearest_other_centroid_similarity": sum(nearest_other_sims) / len(nearest_other_sims),
        "centroid_margin": (sum(own_sims) - sum(nearest_other_sims)) / len(own_sims),
    }


def save_metrics(output_dir: Path, metrics: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    fieldnames = list(metrics[0].keys())
    with open(output_dir / "metrics.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = load_config(args.config)

    baseline_model, preprocess_val = load_model(config, checkpoint=None, device=device)
    dataset = ImageFolder(root=args.data_root, transform=preprocess_val)
    gallery_indices, query_indices, per_class_counts = build_split(
        dataset=dataset,
        gallery_per_class=args.gallery_per_class,
        query_per_class=args.query_per_class,
        max_classes=args.max_classes,
        seed=args.seed,
    )

    print(f"classes={len(per_class_counts)} gallery={len(gallery_indices)} query={len(query_indices)}")
    print(f"device={device} batch_size={args.batch_size}")

    base_gallery_features, gallery_labels = extract_features(
        baseline_model,
        dataset,
        gallery_indices,
        args.batch_size,
        args.num_workers,
        device,
        "baseline gallery",
    )
    base_query_features, query_labels = extract_features(
        baseline_model,
        dataset,
        query_indices,
        args.batch_size,
        args.num_workers,
        device,
        "baseline query",
    )
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    student_model, _ = load_model(config, checkpoint=args.checkpoint, device=device)
    student_gallery_features, student_gallery_labels = extract_features(
        student_model,
        dataset,
        gallery_indices,
        args.batch_size,
        args.num_workers,
        device,
        "student gallery",
    )
    student_query_features, student_query_labels = extract_features(
        student_model,
        dataset,
        query_indices,
        args.batch_size,
        args.num_workers,
        device,
        "student query",
    )
    del student_model

    baseline_metrics = {
        "model": "openclip_baseline",
        **knn_metrics(base_gallery_features, gallery_labels, base_query_features, query_labels, device),
        **centroid_metrics(base_query_features, query_labels),
    }
    student_metrics = {
        "model": Path(args.checkpoint).stem,
        **knn_metrics(
            student_gallery_features,
            student_gallery_labels,
            student_query_features,
            student_query_labels,
            device,
        ),
        **centroid_metrics(student_query_features, student_query_labels),
    }

    same_image_cosine = (base_query_features * student_query_features).sum(dim=1)
    drift_metrics = {
        "model": "baseline_vs_atas_query_drift",
        "top1_knn": "",
        "top5_neighbor_hit": "",
        "mean_top1_similarity": "",
        "mean_own_centroid_similarity": float(same_image_cosine.mean()),
        "mean_nearest_other_centroid_similarity": float(same_image_cosine.min()),
        "centroid_margin": float(same_image_cosine.max()),
    }

    metrics = [baseline_metrics, student_metrics, drift_metrics]
    save_metrics(Path(args.output_dir), metrics)

    for row in metrics:
        print(json.dumps(row, ensure_ascii=False, indent=2))
    print(f"saved={Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
