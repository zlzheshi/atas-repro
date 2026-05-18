from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.transforms import InterpolationMode, Normalize, ToTensor
from torchvision.transforms import functional as TF
from tqdm import tqdm

from train_atas import encode_visual_tokens


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-name", default="openclip_baseline")
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--background-class", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--image-dir", default="JPEGImages")
    parser.add_argument("--mask-dir", default="SegmentationClass")
    parser.add_argument("--split-dir", default="ImageSets/Segmentation")
    parser.add_argument("--output-dir", default="outputs/voc_like_zero_shot_seg")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--save-vis", type=int, default=12)
    parser.add_argument("--save-vis-per-class", type=int, default=0)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_ids(data_root: Path, split_dir: str, split: str) -> list[str]:
    split_file = data_root / split_dir / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"split file not found: {split_file}")
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_image(path: Path) -> Path:
    for suffix in [".jpg", ".jpeg", ".png", ".bmp"]:
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"image not found for stem: {path}")


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


@torch.no_grad()
def encode_text_features(model: torch.nn.Module, classes: list[str], device: torch.device) -> torch.Tensor:
    templates = [
        "a photo of {}.",
        "a close-up photo of {}.",
        "a cropped photo of {}.",
        "a photo of a surface with {}.",
        "a photo of a defect: {}.",
        "an industrial image showing {}.",
    ]
    features = []
    for class_name in classes:
        prompts = [template.format(class_name) for template in templates]
        tokens = open_clip.tokenize(prompts).to(device)
        text = model.encode_text(tokens).float()
        text = F.normalize(text, dim=-1)
        features.append(F.normalize(text.mean(dim=0), dim=0))
    return torch.stack(features)


def preprocess_pair(image_path: Path, mask_path: Path, image_size: int) -> tuple[torch.Tensor, torch.Tensor, Image.Image]:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path)
    image = TF.resize(image, [image_size, image_size], interpolation=InterpolationMode.BICUBIC)
    mask = TF.resize(mask, [image_size, image_size], interpolation=InterpolationMode.NEAREST)
    display = image.copy()
    tensor = Normalize(mean=CLIP_MEAN, std=CLIP_STD)(ToTensor()(image))
    target = torch.as_tensor(list(mask.getdata()), dtype=torch.long).reshape(image_size, image_size)
    return tensor, target, display


@torch.no_grad()
def predict_mask(
    model: torch.nn.Module,
    image: torch.Tensor,
    text_features: torch.Tensor,
    device: torch.device,
    temperature: float,
    label_offset: int,
) -> torch.Tensor:
    image = image.unsqueeze(0).to(device)
    _, patches = encode_visual_tokens(model.visual, image)
    patches = F.normalize(patches.float(), dim=-1)
    logits = patches @ text_features.t()
    logits = logits / temperature
    grid = int(logits.shape[1] ** 0.5)
    logits = logits.reshape(1, grid, grid, text_features.shape[0]).permute(0, 3, 1, 2)
    logits = F.interpolate(logits, size=(image.shape[-2], image.shape[-1]), mode="bilinear", align_corners=False)
    return logits.argmax(dim=1).squeeze(0).cpu() + label_offset


def update_confusion(confusion: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, label_offset: int) -> None:
    num_classes = confusion.shape[0]
    if label_offset == 0:
        valid = (target >= 0) & (target < num_classes)
    else:
        valid = (target >= 1) & (target <= num_classes)
    if valid.sum() == 0:
        return
    pred = pred[valid].clamp(label_offset, num_classes + label_offset - 1) - label_offset
    target = target[valid] - label_offset
    index = target * num_classes + pred
    confusion += torch.bincount(index, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def compute_metrics(confusion: torch.Tensor, classes: list[str], has_background: bool) -> dict[str, object]:
    confusion = confusion.float()
    tp = confusion.diag()
    row_sum = confusion.sum(dim=1)
    col_sum = confusion.sum(dim=0)
    union = row_sum + col_sum - tp
    iou = tp / union.clamp_min(1)
    present = row_sum > 0
    pixel_acc = tp.sum() / confusion.sum().clamp_min(1)
    mean_class_acc = (tp / row_sum.clamp_min(1))[present].mean()
    foreground = present.clone()
    if has_background and foreground.numel() > 0:
        foreground[0] = False
    foreground_pixel_denom = row_sum[foreground].sum().clamp_min(1)
    metrics = {
        "miou": float(iou[present].mean()),
        "pixel_acc": float(pixel_acc),
        "foreground_miou": float(iou[foreground].mean()) if foreground.any() else 0.0,
        "foreground_pixel_acc": float(tp[foreground].sum() / foreground_pixel_denom),
        "mean_class_acc": float(mean_class_acc),
        "class_iou": {classes[i]: float(iou[i]) for i in range(len(classes))},
    }
    return metrics


def mask_to_rgb(mask: torch.Tensor, num_classes: int, label_offset: int) -> Image.Image:
    foreground_palette = [
        (220, 20, 60),
        (65, 105, 225),
        (255, 165, 0),
        (50, 205, 50),
        (148, 0, 211),
        (0, 206, 209),
    ]
    palette = [
        (35, 35, 35),
        *foreground_palette,
    ] if label_offset == 0 else foreground_palette
    rgb = torch.zeros(mask.shape[0], mask.shape[1], 3, dtype=torch.uint8)
    for class_index in range(num_classes):
        label = class_index + label_offset
        rgb[mask == label] = torch.tensor(palette[class_index % len(palette)], dtype=torch.uint8)
    return Image.fromarray(rgb.numpy(), mode="RGB")


def save_vis(
    output_dir: Path,
    image_id: str,
    display: Image.Image,
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    label_offset: int,
) -> None:
    target_img = mask_to_rgb(target, num_classes, label_offset)
    pred_img = mask_to_rgb(pred, num_classes, label_offset)
    overlay = Image.blend(display, pred_img, alpha=0.45)
    gap = 8
    width = display.width * 4 + gap * 3
    canvas = Image.new("RGB", (width, display.height), "white")
    x = 0
    for panel in [display, target_img, pred_img, overlay]:
        canvas.paste(panel, (x, 0))
        x += display.width + gap
    canvas.save(output_dir / f"{image_id}_seg.png")


def save_outputs(output_dir: Path, model_name: str, metrics: dict[str, object], confusion: torch.Tensor) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump({"model": model_name, **metrics}, file, ensure_ascii=False, indent=2)
    with open(output_dir / "class_iou.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "iou"])
        for name, iou in metrics["class_iou"].items():
            writer.writerow([name, iou])
    torch.save(confusion, output_dir / "confusion.pt")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    ids = load_ids(data_root, args.split_dir, args.split)
    if args.limit is not None:
        ids = ids[: args.limit]
    classes = ([args.background_class] if args.background_class is not None else []) + args.classes
    has_background = args.background_class is not None
    label_offset = 0 if has_background else 1
    print(f"model={args.model_name} classes={classes} images={len(ids)} device={device}")

    model = load_model(config, args.checkpoint, device)
    text_features = encode_text_features(model, classes, device)
    confusion = torch.zeros(len(classes), len(classes), dtype=torch.long)
    vis_counts = {class_index: 0 for class_index in range(len(classes))}
    saved_vis_ids: set[str] = set()

    for index, image_id in enumerate(tqdm(ids, desc=args.model_name)):
        image_path = resolve_image(data_root / args.image_dir / image_id)
        mask_path = data_root / args.mask_dir / f"{image_id}.png"
        image, target, display = preprocess_pair(image_path, mask_path, args.image_size)
        pred = predict_mask(model, image, text_features, device, args.temperature, label_offset)
        update_confusion(confusion, pred, target, label_offset)

        target_class_indices = []
        for label in torch.unique(target).tolist():
            label = int(label)
            if label == 255:
                continue
            class_index = label - label_offset
            if 0 <= class_index < len(classes):
                target_class_indices.append(class_index)

        needs_balanced_vis = any(
            (not has_background or class_index != 0) and vis_counts[class_index] < args.save_vis_per_class
            for class_index in target_class_indices
        )
        if image_id not in saved_vis_ids and (index < args.save_vis or needs_balanced_vis):
            save_vis(vis_dir, image_id, display, pred, target, len(classes), label_offset)
            saved_vis_ids.add(image_id)
            for class_index in target_class_indices:
                vis_counts[class_index] += 1

    metrics = compute_metrics(confusion, classes, has_background)
    save_outputs(output_dir, args.model_name, metrics, confusion)
    print(json.dumps({"model": args.model_name, **metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
