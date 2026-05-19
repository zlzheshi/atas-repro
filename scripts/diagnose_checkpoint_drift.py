from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.atas.losses import ATASLossConfig, atas_region_loss
from train_atas import CLIPVisualTokenEncoder, ImagePathFolder, MosaicBatchCollator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose how far ATAS checkpoints drift from the frozen CLIP teacher.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Checkpoint to evaluate. Use NAME=PATH or PATH. Can be repeated.",
    )
    parser.add_argument("--output-dir", default="outputs/checkpoint_drift")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--max-pairwise-patches", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-baseline", action="store_true", help="Do not include the original OpenCLIP baseline.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_checkpoint_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), Path(path.strip())
    path = Path(spec)
    name = path.parent.name + "_" + path.stem
    return name, path


class MetricStore:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = {}

    def add(self, name: str, value: torch.Tensor | float) -> None:
        if isinstance(value, torch.Tensor):
            value = float(value.detach().float().cpu().item())
        self.values.setdefault(name, []).append(float(value))

    def add_tensor(self, prefix: str, tensor: torch.Tensor) -> None:
        tensor = tensor.detach().float()
        self.add(f"{prefix}_mean", tensor.mean())
        self.add(f"{prefix}_std", tensor.std(unbiased=False))
        self.add(f"{prefix}_min", tensor.min())
        self.add(f"{prefix}_max", tensor.max())

    def summary(self) -> dict[str, float]:
        return {
            name: sum(values) / max(len(values), 1)
            for name, values in sorted(self.values.items())
        }


def load_visual_checkpoint(student: nn.Module, checkpoint_path: Path) -> tuple[int, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    visual_state = checkpoint.get("visual_state_dict", checkpoint.get("model", checkpoint))
    missing, unexpected = student.visual.load_state_dict(visual_state, strict=False)
    return len(missing), len(unexpected)


def pairwise_cosine_mse(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    student = F.normalize(student.float(), dim=-1)
    teacher = F.normalize(teacher.float(), dim=-1)
    student_rel = student @ student.t()
    teacher_rel = teacher @ teacher.t()
    return F.mse_loss(student_rel, teacher_rel)


def sample_patch_tokens(
    student_patches: torch.Tensor,
    teacher_patches: torch.Tensor,
    max_patches: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    student_flat = student_patches.reshape(-1, student_patches.shape[-1])
    teacher_flat = teacher_patches.reshape(-1, teacher_patches.shape[-1])
    if student_flat.shape[0] <= max_patches:
        return student_flat, teacher_flat
    indices = torch.randperm(student_flat.shape[0], generator=generator, device=student_flat.device)[:max_patches]
    return student_flat[indices], teacher_flat[indices]


def make_loader(config: dict, args: argparse.Namespace) -> DataLoader:
    dataset = ImagePathFolder(root=args.data_root)
    batch_size = args.batch_size or int(config["training"]["batch_size"])
    num_workers = args.num_workers
    collator = MosaicBatchCollator(
        global_image_size=int(config["data"]["global_image_size"]),
        mosaic_image_size=int(config["data"]["image_size"]),
        mosaic_choices=list(config["data"]["mosaic_choices"]),
        allow_repeat_fill=bool(config["data"].get("allow_repeat_fill", True)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collator,
    )


def evaluate_model(
    name: str,
    student: nn.Module,
    teacher_encoder: CLIPVisualTokenEncoder,
    loader: DataLoader,
    config: dict,
    args: argparse.Namespace,
) -> dict[str, float | str | int]:
    device = torch.device(args.device)
    student_encoder = CLIPVisualTokenEncoder(student.visual).to(device).eval()
    store = MetricStore()
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    loss_config = ATASLossConfig(
        temperature=float(config["training"]["temperature"]),
        lambda_gld=float(config["training"]["lambda_gld"]),
        lambda_lld=float(config["training"]["lambda_lld"]),
        lambda_ggd=float(config["training"]["lambda_ggd"]),
    )
    max_lld_patches = config["training"].get("max_lld_patches")
    if max_lld_patches is not None:
        max_lld_patches = int(max_lld_patches)

    processed_batches = 0
    processed_images = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.max_batches:
                break
            global_images = batch["global_images"].to(device, non_blocking=True)
            mosaic_images = batch["mosaic_images"].to(device, non_blocking=True)
            region_data = batch["region_boxes"].to(device, non_blocking=True)
            region_boxes = region_data[:, :5]
            source_indices = region_data[:, 5]
            patch_grid = tuple(batch["patch_grid"].tolist())

            teacher_global_cls, teacher_global_patches = teacher_encoder(global_images)
            _, teacher_mosaic_patches = teacher_encoder(mosaic_images)
            student_global_cls, student_global_patches = student_encoder(global_images)
            _, student_mosaic_patches = student_encoder(mosaic_images)

            cls_cos = F.cosine_similarity(student_global_cls.float(), teacher_global_cls.float(), dim=-1)
            global_patch_cos = F.cosine_similarity(
                student_global_patches.reshape(-1, student_global_patches.shape[-1]).float(),
                teacher_global_patches.reshape(-1, teacher_global_patches.shape[-1]).float(),
                dim=-1,
            )
            mosaic_patch_cos = F.cosine_similarity(
                student_mosaic_patches.reshape(-1, student_mosaic_patches.shape[-1]).float(),
                teacher_mosaic_patches.reshape(-1, teacher_mosaic_patches.shape[-1]).float(),
                dim=-1,
            )
            sampled_student, sampled_teacher = sample_patch_tokens(
                student_mosaic_patches,
                teacher_mosaic_patches,
                args.max_pairwise_patches,
                generator,
            )
            loss, metrics = atas_region_loss(
                student_global_cls=student_global_cls,
                student_mosaic_patches=student_mosaic_patches,
                teacher_global_cls=teacher_global_cls,
                teacher_region_cls=teacher_global_cls[source_indices],
                teacher_mosaic_patches=teacher_mosaic_patches,
                region_boxes=region_boxes,
                patch_grid=patch_grid,
                config=loss_config,
                max_lld_patches=max_lld_patches,
            )

            store.add_tensor("cls_cos_to_teacher", cls_cos)
            store.add_tensor("global_patch_cos_to_teacher", global_patch_cos)
            store.add_tensor("mosaic_patch_cos_to_teacher", mosaic_patch_cos)
            store.add("cls_pairwise_mse", pairwise_cosine_mse(student_global_cls, teacher_global_cls))
            store.add("mosaic_patch_pairwise_mse", pairwise_cosine_mse(sampled_student, sampled_teacher))
            store.add("loss", loss)
            for metric_name, value in metrics.items():
                store.add(metric_name, value)

            processed_batches += 1
            processed_images += int(global_images.shape[0])

    result: dict[str, float | str | int] = {
        "name": name,
        "batches": processed_batches,
        "images": processed_images,
    }
    result.update(store.summary())
    return result


def write_outputs(results: list[dict[str, float | str | int]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    fieldnames = sorted({key for result in results for key in result.keys()})
    ordered = ["name", "checkpoint", "batches", "images", "missing_keys", "unexpected_keys"]
    fieldnames = [key for key in ordered if key in fieldnames] + [key for key in fieldnames if key not in ordered]
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    columns = [
        "name",
        "cls_cos_to_teacher_mean",
        "global_patch_cos_to_teacher_mean",
        "mosaic_patch_cos_to_teacher_mean",
        "cls_pairwise_mse",
        "mosaic_patch_pairwise_mse",
        "loss_gld",
        "loss_lld",
        "loss_ggd",
    ]
    with (output_dir / "metrics.md").open("w", encoding="utf-8") as file:
        file.write("# Checkpoint Drift Diagnostics\n\n")
        file.write("| " + " | ".join(columns) + " |\n")
        file.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for result in results:
            row = []
            for column in columns:
                value = result.get(column, "")
                if isinstance(value, float):
                    row.append(f"{value:.6f}")
                else:
                    row.append(str(value))
            file.write("| " + " | ".join(row) + " |\n")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    config = load_config(args.config)
    device = torch.device(args.device)

    model_name = config["model"]["name"]
    pretrained = config["model"]["pretrained"]
    base_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    base_model = base_model.eval()
    teacher = copy.deepcopy(base_model).to(device).eval()
    teacher_encoder = CLIPVisualTokenEncoder(teacher.visual).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    loader = make_loader(config, args)
    specs: list[tuple[str, Path | None]] = []
    if not args.no_baseline:
        specs.append(("baseline", None))
    specs.extend((name, path) for name, path in map(parse_checkpoint_spec, args.checkpoint))

    results: list[dict[str, float | str | int]] = []
    for name, checkpoint_path in specs:
        student = copy.deepcopy(base_model).to(device).eval()
        result_meta: dict[str, float | str | int] = {
            "checkpoint": "baseline" if checkpoint_path is None else str(checkpoint_path),
            "missing_keys": 0,
            "unexpected_keys": 0,
        }
        if checkpoint_path is not None:
            missing, unexpected = load_visual_checkpoint(student, checkpoint_path)
            result_meta["missing_keys"] = missing
            result_meta["unexpected_keys"] = unexpected
        result = evaluate_model(name, student, teacher_encoder, loader, config, args)
        result.update(result_meta)
        results.append(result)
        del student
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_outputs(results, Path(args.output_dir))
    print(f"wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
