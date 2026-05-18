from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode, Normalize, ToTensor
from torchvision.transforms import functional as TF

from train_atas import encode_visual_tokens


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/patch_alignment_vis")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class-filter", nargs="*", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--alpha", type=float, default=0.52)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_model(config: dict, checkpoint: str | None, device: torch.device) -> torch.nn.Module:
    model, _, _ = open_clip.create_model_and_transforms(
        config["model"]["name"],
        pretrained=config["model"]["pretrained"],
    )
    if checkpoint is not None:
        loaded = torch.load(checkpoint, map_location="cpu")
        visual_state = loaded.get("visual_state_dict", loaded.get("model", loaded))
        missing, unexpected = model.visual.load_state_dict(visual_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint load mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
            )
    return model.to(device).eval()


def preprocess_image(path: str, image_size: int) -> tuple[torch.Tensor, Image.Image]:
    image = Image.open(path).convert("RGB")
    image = TF.resize(image, image_size, interpolation=InterpolationMode.BICUBIC)
    image = TF.center_crop(image, [image_size, image_size])
    display = image.copy()
    tensor = ToTensor()(image)
    tensor = Normalize(mean=CLIP_MEAN, std=CLIP_STD)(tensor)
    return tensor, display


def choose_samples(
    dataset: ImageFolder,
    num_images: int,
    seed: int,
    class_filter: list[str] | None,
) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    class_to_paths: dict[int, list[str]] = {}
    allowed = set(class_filter) if class_filter else None

    for path, label in dataset.samples:
        class_name = dataset.classes[label]
        if allowed is not None and class_name not in allowed:
            continue
        class_to_paths.setdefault(label, []).append(path)

    labels = sorted(class_to_paths)
    rng.shuffle(labels)

    samples: list[tuple[str, int]] = []
    for label in labels:
        paths = class_to_paths[label]
        rng.shuffle(paths)
        samples.append((paths[0], label))
        if len(samples) >= num_images:
            break

    if len(samples) < num_images:
        raise ValueError(f"only found {len(samples)} samples, but {num_images} requested")
    return samples


@torch.no_grad()
def compute_patch_map(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    prompt: str | None,
) -> np.ndarray:
    image = image.unsqueeze(0).to(device)
    cls, patches = encode_visual_tokens(model.visual, image)
    patches = F.normalize(patches.float(), dim=-1)

    if prompt is None:
        query = F.normalize(cls.float(), dim=-1)
    else:
        tokens = open_clip.tokenize([prompt]).to(device)
        query = F.normalize(model.encode_text(tokens).float(), dim=-1)

    sim = (patches @ query[:, :, None]).squeeze(0).squeeze(-1)
    grid_size = int(math.sqrt(sim.numel()))
    sim = sim.reshape(grid_size, grid_size)
    sim = sim - sim.min()
    sim = sim / (sim.max().clamp_min(1e-6))
    return sim.detach().cpu().numpy()


def colorize_heatmap(values: np.ndarray) -> Image.Image:
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(1.7 * values - 0.25, 0.0, 1.0)
    green = np.clip(1.7 * (1.0 - np.abs(values - 0.5) * 2.0), 0.0, 1.0)
    blue = np.clip(1.4 * (1.0 - values) - 0.1, 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float) -> Image.Image:
    heat = colorize_heatmap(heatmap).resize(image.size, resample=Image.Resampling.BICUBIC)
    return Image.blend(image.convert("RGB"), heat, alpha=alpha)


def add_caption(image: Image.Image, caption: str, height: int = 34) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + height), "white")
    canvas.paste(image, (0, height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 9), caption, fill=(20, 20, 20))
    return canvas


def make_panel(
    original: Image.Image,
    baseline_overlay: Image.Image,
    atas_overlay: Image.Image,
    delta_overlay: Image.Image,
    title: str,
) -> Image.Image:
    panels = [
        add_caption(original, "input"),
        add_caption(baseline_overlay, "OpenCLIP"),
        add_caption(atas_overlay, "ATAS"),
        add_caption(delta_overlay, "ATAS - OpenCLIP"),
    ]
    gap = 8
    title_h = 42
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels) + title_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 12), title, fill=(0, 0, 0))

    x = 0
    for panel in panels:
        canvas.paste(panel, (x, title_h))
        x += panel.width + gap
    return canvas


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ImageFolder(args.data_root)
    samples = choose_samples(dataset, args.num_images, args.seed, args.class_filter)

    baseline = load_model(config, checkpoint=None, device=device)
    atas = load_model(config, checkpoint=args.checkpoint, device=device)

    metadata = []
    for index, (path, label) in enumerate(samples):
        tensor, display = preprocess_image(path, args.image_size)
        base_map = compute_patch_map(baseline, tensor, device, args.prompt)
        atas_map = compute_patch_map(atas, tensor, device, args.prompt)
        delta = atas_map - base_map
        delta = delta - delta.min()
        delta = delta / max(float(delta.max()), 1e-6)

        baseline_overlay = overlay_heatmap(display, base_map, args.alpha)
        atas_overlay = overlay_heatmap(display, atas_map, args.alpha)
        delta_overlay = overlay_heatmap(display, delta, args.alpha)

        class_name = dataset.classes[label]
        query = args.prompt if args.prompt else "image CLS"
        title = f"{index:02d}  class={class_name}  query={query}"
        panel = make_panel(display, baseline_overlay, atas_overlay, delta_overlay, title)
        out_name = f"patch_alignment_{index:02d}_{class_name}.png"
        panel.save(output_dir / out_name)

        metadata.append(
            {
                "file": out_name,
                "source_image": path,
                "class": class_name,
                "query": query,
                "baseline_mean": float(base_map.mean()),
                "atas_mean": float(atas_map.mean()),
                "delta_mean": float((atas_map - base_map).mean()),
            }
        )
        print(f"saved {output_dir / out_name}")

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    with open(output_dir / "README.md", "w", encoding="utf-8") as file:
        file.write("# Patch 级对齐可视化\n\n")
        file.write(f"- 检查点：`{args.checkpoint}`\n")
        file.write(f"- 查询语义：`{args.prompt if args.prompt else 'image CLS'}`\n")
        file.write(f"- 输入分辨率：`{args.image_size}`\n\n")
        for item in metadata:
            file.write(f"![{item['class']}]({item['file']})\n\n")


if __name__ == "__main__":
    main()
