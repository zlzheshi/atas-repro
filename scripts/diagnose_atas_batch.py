from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.atas.losses import contrastive_self_distill, weighted_region_pool
from train_atas import CLIPVisualTokenEncoder, ImagePathFolder, MosaicBatchCollator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def summarize_tensor(name: str, tensor: torch.Tensor) -> None:
    tensor = tensor.detach().float()
    finite = torch.isfinite(tensor)
    print(
        f"{name}: shape={tuple(tensor.shape)} "
        f"finite={finite.float().mean().item():.3f} "
        f"min={tensor.min().item():.6f} mean={tensor.mean().item():.6f} "
        f"max={tensor.max().item():.6f} std={tensor.std().item():.6f}"
    )


def summarize_pairwise(name: str, student: torch.Tensor, teacher: torch.Tensor, temperature: float) -> None:
    student = F.normalize(student.float(), dim=-1)
    teacher = F.normalize(teacher.float(), dim=-1)
    logits = student @ teacher.t()
    diag = logits.diag()
    mask = ~torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    offdiag = logits[mask]
    loss = contrastive_self_distill(student, teacher, temperature)
    print(
        f"{name}: loss={loss.item():.6f} diag_mean={diag.mean().item():.6f} "
        f"diag_min={diag.min().item():.6f} offdiag_mean={offdiag.mean().item():.6f} "
        f"offdiag_max={offdiag.max().item():.6f}"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)

    torch.manual_seed(0)
    model_name = config["model"]["name"]
    pretrained = config["model"]["pretrained"]
    student, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    teacher = copy.deepcopy(student)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        visual_state = checkpoint.get("visual_state_dict", checkpoint.get("model", checkpoint))
        missing, unexpected = student.visual.load_state_dict(visual_state, strict=False)
        print(f"loaded_checkpoint={args.checkpoint}")
        print(f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    student_encoder = CLIPVisualTokenEncoder(student.visual).to(device).eval()
    teacher_encoder = CLIPVisualTokenEncoder(teacher.visual).to(device).eval()
    model = student.to(device).eval()

    dataset = ImagePathFolder(root=args.data_root)
    collator = MosaicBatchCollator(
        global_image_size=config["data"]["global_image_size"],
        mosaic_image_size=config["data"]["image_size"],
        mosaic_choices=config["data"]["mosaic_choices"],
        allow_repeat_fill=bool(config["data"].get("allow_repeat_fill", True)),
    )
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=collator,
    )
    batch = next(iter(loader))
    global_images = batch["global_images"].to(device)
    mosaic_images = batch["mosaic_images"].to(device)
    region_data = batch["region_boxes"].to(device)
    region_boxes = region_data[:, :5]
    source_indices = region_data[:, 5]
    patch_grid = tuple(batch["patch_grid"].tolist())
    temperature = float(config["training"]["temperature"])

    print(f"config={Path(args.config).name}")
    print(f"global_images={tuple(global_images.shape)} mosaic_images={tuple(mosaic_images.shape)}")
    print(f"grid={int(batch['mosaic_grid'].item())} unique={int(batch['unique_sources'].item())}/{region_data.shape[0]}")

    with torch.no_grad():
        teacher_global_cls, _ = teacher_encoder(global_images)
        student_global_cls, _ = student_encoder(global_images)
        _, teacher_mosaic_patches = teacher_encoder(mosaic_images)
        _, student_mosaic_patches = student_encoder(mosaic_images)
        openclip_image_features = model.encode_image(global_images)

    summarize_tensor("student_global_cls_norm", student_global_cls.norm(dim=-1))
    summarize_tensor("teacher_global_cls_norm", teacher_global_cls.norm(dim=-1))
    summarize_tensor("student_teacher_abs_diff", (student_global_cls - teacher_global_cls).abs())
    summarize_pairwise("manual_cls_vs_manual_cls", student_global_cls, teacher_global_cls, temperature)
    summarize_pairwise("openclip_encode_image_self", openclip_image_features, openclip_image_features, temperature)

    teacher_region_cls = teacher_global_cls[source_indices]
    pooled_regions = weighted_region_pool(
        student_patches=student_mosaic_patches,
        teacher_cls=teacher_region_cls,
        region_boxes=region_boxes,
        patch_grid=patch_grid,
        temperature=temperature,
    )
    summarize_tensor("student_mosaic_patch_norm", student_mosaic_patches.norm(dim=-1))
    summarize_tensor("pooled_region_norm", pooled_regions.norm(dim=-1))
    summarize_pairwise("pooled_region_vs_teacher_region", pooled_regions, teacher_region_cls, temperature)


if __name__ == "__main__":
    main()
