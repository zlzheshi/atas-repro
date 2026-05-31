from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path

import open_clip
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, InterpolationMode, Lambda, Normalize, ToTensor
from torchvision.transforms import functional as TF
from tqdm import tqdm

from src.atas.losses import ATASLossConfig, atas_region_loss
from src.atas.mosaic import build_mosaic, choose_mosaic_grid


class ImagePathFolder(Dataset):
    def __init__(self, root: str) -> None:
        self.dataset = ImageFolder(root=root)

    def __len__(self) -> int:
        return len(self.dataset.samples)

    def __getitem__(self, index: int) -> str:
        return self.dataset.samples[index][0]


class MosaicBatchCollator:
    def __init__(
        self,
        global_image_size: int,
        mosaic_image_size: int,
        mosaic_choices: list[int],
        allow_repeat_fill: bool = True,
    ) -> None:
        self.global_image_size = global_image_size
        self.mosaic_image_size = mosaic_image_size
        self.mosaic_choices = mosaic_choices
        self.allow_repeat_fill = allow_repeat_fill
        self.normalize = Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
        self.to_tensor = Compose(
            [
                Lambda(lambda image: image.convert("RGB")),
                ToTensor(),
                self.normalize,
            ]
        )

    def _load_image(self, path: str) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def __call__(self, paths: list[str]) -> dict[str, torch.Tensor]:
        images = [self._load_image(path) for path in paths]
        grid = choose_mosaic_grid(self.mosaic_choices)
        images_per_mosaic = grid * grid
        num_mosaics = math.ceil(len(images) / images_per_mosaic)

        global_images = [
            self.to_tensor(
                TF.resize(image, [self.global_image_size, self.global_image_size], interpolation=InterpolationMode.BICUBIC)
            )
            for image in images
        ]

        mosaic_images = []
        region_boxes = []
        patch_grid_size = self.mosaic_image_size // 16
        cell_patch_size = patch_grid_size // grid

        for mosaic_index in range(num_mosaics):
            start = mosaic_index * images_per_mosaic
            chunk = images[start : start + images_per_mosaic]
            if len(chunk) < images_per_mosaic:
                if not self.allow_repeat_fill:
                    raise ValueError(
                        f"mosaic grid {grid} requires chunks of {images_per_mosaic} images, "
                        f"but the last chunk only has {len(chunk)}. Increase batch_size, "
                        "use a divisor-compatible mosaic_choices value, or enable allow_repeat_fill."
                    )
                pad_count = images_per_mosaic - len(chunk)
                repeats = math.ceil(pad_count / len(images))
                padding = (images * repeats)[:pad_count]
                chunk = chunk + padding

            mosaic = build_mosaic(chunk, grid_size=grid, output_size=self.mosaic_image_size)
            mosaic_images.append(self.to_tensor(mosaic))

            for local_index in range(images_per_mosaic):
                source_index = (start + local_index) % len(images)
                row, col = divmod(local_index, grid)
                region_boxes.append(
                    [
                        mosaic_index,
                        row * cell_patch_size,
                        (row + 1) * cell_patch_size,
                        col * cell_patch_size,
                        (col + 1) * cell_patch_size,
                        source_index,
                    ]
                )

        return {
            "global_images": torch.stack(global_images),
            "mosaic_images": torch.stack(mosaic_images),
            "region_boxes": torch.tensor(region_boxes, dtype=torch.long),
            "patch_grid": torch.tensor([patch_grid_size, patch_grid_size], dtype=torch.long),
            "mosaic_grid": torch.tensor(grid, dtype=torch.long),
            "unique_sources": torch.tensor(len({box[-1] for box in region_boxes}), dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/atas_vitb.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resize_positional_embedding(positional_embedding: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
    cls_pos = positional_embedding[:1]
    patch_pos = positional_embedding[1:]
    old_size = int(math.sqrt(patch_pos.shape[0]))
    if (old_size, old_size) == grid_size:
        return positional_embedding
    patch_pos = patch_pos.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=grid_size, mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(grid_size[0] * grid_size[1], -1)
    return torch.cat([cls_pos, patch_pos], dim=0)


def unwrap_visual(visual: nn.Module) -> nn.Module:
    return visual.module if hasattr(visual, "module") else visual


def encode_visual_tokens(visual: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract CLS and patch tokens from an OpenCLIP visual transformer.

    This follows OpenCLIP's ViT path and keeps the implementation explicit so the
    ATAS losses can consume both global and local representations.
    """
    visual = unwrap_visual(visual)
    x = visual.conv1(images)
    grid_size = (x.shape[2], x.shape[3])
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)

    cls_embedding = visual.class_embedding.to(x.dtype)
    cls_tokens = cls_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls_tokens, x], dim=1)
    positional_embedding = resize_positional_embedding(visual.positional_embedding, grid_size)
    x = x + positional_embedding.to(dtype=x.dtype, device=x.device)
    if hasattr(visual, "patch_dropout"):
        x = visual.patch_dropout(x)

    x = visual.ln_pre(x)
    x = visual.transformer(x)
    x = visual.ln_post(x)

    if visual.proj is not None:
        x = x @ visual.proj

    pool_type = getattr(visual, "pool_type", "tok")
    if pool_type == "avg":
        return x[:, 1:].mean(dim=1), x[:, 1:]
    if pool_type == "tok":
        return x[:, 0], x[:, 1:]
    return x, x


def encode_image_tokens(model: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return encode_visual_tokens(model.visual, images)


class CLIPVisualTokenEncoder(nn.Module):
    def __init__(self, visual: nn.Module) -> None:
        super().__init__()
        self.visual = visual

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return encode_visual_tokens(self.visual, images)


def unwrap_encoder(encoder: nn.Module) -> CLIPVisualTokenEncoder:
    return encoder.module if hasattr(encoder, "module") else encoder


def init_distributed() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def is_main_process(rank: int) -> bool:
    return rank == 0


def load_checkpoint(
    path: str,
    student_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    visual_state = checkpoint.get("visual_state_dict", checkpoint.get("model", checkpoint))
    unwrap_visual(unwrap_encoder(student_encoder).visual).load_state_dict(visual_state, strict=False)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("epoch", 0))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    distributed, rank, _, local_rank = init_distributed()
    if distributed:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model_name = config["model"]["name"]
    pretrained = config["model"]["pretrained"]
    force_quick_gelu = bool(config["model"].get("quick_gelu", False))
    student, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        force_quick_gelu=force_quick_gelu,
    )
    teacher = copy.deepcopy(student)
    teacher.eval()

    student_encoder: nn.Module = CLIPVisualTokenEncoder(student.visual).to(device)
    teacher_encoder = CLIPVisualTokenEncoder(teacher.visual).to(device)
    teacher_encoder.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    if distributed:
        student_encoder = DistributedDataParallel(student_encoder, device_ids=[local_rank], output_device=local_rank)

    dataset = ImagePathFolder(root=args.data_root)
    collator = MosaicBatchCollator(
        global_image_size=config["data"]["global_image_size"],
        mosaic_image_size=config["data"]["image_size"],
        mosaic_choices=config["data"]["mosaic_choices"],
        allow_repeat_fill=bool(config["data"].get("allow_repeat_fill", True)),
    )
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
    )

    loss_config = ATASLossConfig(
        temperature=config["training"]["temperature"],
        lambda_gld=config["training"]["lambda_gld"],
        lambda_lld=config["training"]["lambda_lld"],
        lambda_ggd=config["training"]["lambda_ggd"],
    )
    optimizer = torch.optim.AdamW(
        student_encoder.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    output_dir = Path(config["training"]["output_dir"])
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)

    use_amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))
    start_epoch = 0
    if args.resume is not None:
        start_epoch = load_checkpoint(args.resume, student_encoder, optimizer, scaler)

    for epoch in range(start_epoch, config["training"]["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)
        student_encoder.train()
        progress = tqdm(loader, desc=f"epoch {epoch + 1}", disable=not is_main_process(rank))
        optimizer.zero_grad(set_to_none=True)
        max_steps = config["training"].get("max_steps")
        for step, batch in enumerate(progress):
            global_images = batch["global_images"].to(device, non_blocking=True)
            mosaic_images = batch["mosaic_images"].to(device, non_blocking=True)
            region_data = batch["region_boxes"].to(device, non_blocking=True)
            patch_grid = tuple(batch["patch_grid"].tolist())
            region_boxes = region_data[:, :5]
            source_indices = region_data[:, 5]

            with torch.no_grad():
                teacher_global_cls, _ = teacher_encoder(global_images)
                _, teacher_mosaic_patches = teacher_encoder(mosaic_images)

            with torch.cuda.amp.autocast(enabled=use_amp):
                student_global_cls, _ = student_encoder(global_images)
                _, student_mosaic_patches = student_encoder(mosaic_images)
                loss, metrics = atas_region_loss(
                    student_global_cls=student_global_cls,
                    student_mosaic_patches=student_mosaic_patches,
                    teacher_global_cls=teacher_global_cls,
                    teacher_region_cls=teacher_global_cls[source_indices],
                    teacher_mosaic_patches=teacher_mosaic_patches,
                    region_boxes=region_boxes,
                    patch_grid=patch_grid,
                    config=loss_config,
                    max_lld_patches=config["training"]["max_lld_patches"],
                )

            scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if is_main_process(rank) and step % config["training"]["log_interval"] == 0:
                postfix = {key: f"{value.item():.4f}" for key, value in metrics.items()}
                postfix["grid"] = int(batch["mosaic_grid"].item())
                postfix["unique"] = f"{int(batch['unique_sources'].item())}/{int(region_data.shape[0])}"
                progress.set_postfix(postfix)

            if max_steps is not None and step + 1 >= int(max_steps):
                break

        if is_main_process(rank) and (epoch + 1) % int(config["training"].get("save_interval", 1)) == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "visual_state_dict": unwrap_visual(unwrap_encoder(student_encoder).visual).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
            }
            torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch + 1}.pt")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
