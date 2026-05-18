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
from torch import nn
from torchvision.transforms import InterpolationMode, Normalize, ToTensor
from torchvision.transforms import functional as TF
from tqdm import tqdm

from train_atas import encode_visual_tokens, resize_positional_embedding


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv monitor",
]

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--voc-root", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-name", default="openclip_baseline")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", default="outputs/voc_zero_shot_seg")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--save-vis", type=int, default=12)
    parser.add_argument(
        "--dense-mode",
        choices=["vanilla", "maskclip", "sclip"],
        default="vanilla",
        help="Dense visual feature extraction mode. vanilla uses final ViT patch tokens; "
        "maskclip uses value embeddings from the last self-attention block; "
        "sclip uses self-correlation attention in the last block.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def voc_paths(voc_root: Path, split: str) -> list[tuple[Path, Path, str]]:
    image_set = voc_root / "ImageSets" / "Segmentation" / f"{split}.txt"
    if not image_set.exists():
        raise FileNotFoundError(f"VOC split file not found: {image_set}")

    ids = [line.strip() for line in image_set.read_text(encoding="utf-8").splitlines() if line.strip()]
    paths = []
    for image_id in ids:
        image_path = voc_root / "JPEGImages" / f"{image_id}.jpg"
        mask_path = voc_root / "SegmentationClass" / f"{image_id}.png"
        if mask_path.exists():
            paths.append((image_path, mask_path, image_id))
    return paths


def load_model(config: dict, checkpoint: str | None, device: torch.device) -> nn.Module:
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
def encode_text_features(model: nn.Module, device: torch.device) -> torch.Tensor:
    templates = [
        "a photo of a {}.",
        "a photo of the {}.",
        "a close-up photo of a {}.",
        "a cropped photo of a {}.",
        "a photo of a small {}.",
        "a photo of a large {}.",
    ]
    class_features = []
    for class_name in VOC_CLASSES:
        prompts = [template.format(class_name) for template in templates]
        tokens = open_clip.tokenize(prompts).to(device)
        features = model.encode_text(tokens).float()
        features = F.normalize(features, dim=-1)
        class_features.append(F.normalize(features.mean(dim=0), dim=0))
    return torch.stack(class_features, dim=0)


def preprocess_pair(image_path: Path, mask_path: Path, image_size: int) -> tuple[torch.Tensor, torch.Tensor, Image.Image]:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path)

    image = TF.resize(image, [image_size, image_size], interpolation=InterpolationMode.BICUBIC)
    mask = TF.resize(mask, [image_size, image_size], interpolation=InterpolationMode.NEAREST)
    display = image.copy()

    tensor = ToTensor()(image)
    tensor = Normalize(mean=CLIP_MEAN, std=CLIP_STD)(tensor)
    target = torch.as_tensor(list(mask.getdata()), dtype=torch.long).reshape(image_size, image_size)
    return tensor, target, display


@torch.no_grad()
def encode_maskclip_value_tokens(visual: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Approximate MaskCLIP dense features from the last attention value branch.

    MaskCLIP-style zero-shot segmentation extracts value embeddings from the
    final self-attention block instead of using the final CLS-pooled CLIP output.
    This keeps more spatial detail for patch-text matching.
    """
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
    blocks = visual.transformer.resblocks
    for block in blocks[:-1]:
        x = block(x)

    last = blocks[-1]
    value_input = last.ln_1(x)
    embed_dim = last.attn.embed_dim
    value_weight = last.attn.in_proj_weight[2 * embed_dim :]
    value_bias = None
    if last.attn.in_proj_bias is not None:
        value_bias = last.attn.in_proj_bias[2 * embed_dim :]
    values = F.linear(value_input, value_weight, value_bias)
    values = visual.ln_post(values)

    if visual.proj is not None:
        values = values @ visual.proj
    return values[:, 1:]


def apply_layer_scale(block: nn.Module, name: str, x: torch.Tensor) -> torch.Tensor:
    layer_scale = getattr(block, name, None)
    if layer_scale is None:
        return x
    return layer_scale(x)


def self_correlation_attention(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    attn = block.attn
    qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)

    batch_size, token_count, embed_dim = q.shape
    num_heads = attn.num_heads
    head_dim = embed_dim // num_heads

    def reshape_heads(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.reshape(batch_size, token_count, num_heads, head_dim)
        return tensor.permute(0, 2, 1, 3).reshape(batch_size * num_heads, token_count, head_dim)

    q = reshape_heads(q)
    k = reshape_heads(k)
    v = reshape_heads(v)

    scale = head_dim ** -0.5
    q_attn = torch.softmax(torch.bmm(q, q.transpose(1, 2)) * scale, dim=-1)
    k_attn = torch.softmax(torch.bmm(k, k.transpose(1, 2)) * scale, dim=-1)
    out = torch.bmm(q_attn + k_attn, v)
    out = out.reshape(batch_size, num_heads, token_count, head_dim)
    out = out.permute(0, 2, 1, 3).reshape(batch_size, token_count, embed_dim)
    return attn.out_proj(out)


@torch.no_grad()
def encode_sclip_tokens(visual: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Approximate SCLIP dense features using self-correlation attention.

    SCLIP replaces the final ViT block attention map with self-correlation
    attention to improve dense patch-text alignment while keeping CLIP weights.
    """
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
    blocks = visual.transformer.resblocks
    for block in blocks[:-1]:
        x = block(x)

    last = blocks[-1]
    attn_out = self_correlation_attention(last, last.ln_1(x))
    x = x + apply_layer_scale(last, "ls_1", attn_out)
    x = x + apply_layer_scale(last, "ls_2", last.mlp(last.ln_2(x)))
    x = visual.ln_post(x)

    if visual.proj is not None:
        x = x @ visual.proj
    return x[:, 1:]


@torch.no_grad()
def predict_mask(
    model: nn.Module,
    image: torch.Tensor,
    text_features: torch.Tensor,
    device: torch.device,
    temperature: float,
    dense_mode: str,
) -> torch.Tensor:
    image = image.unsqueeze(0).to(device)
    if dense_mode == "maskclip":
        patches = encode_maskclip_value_tokens(model.visual, image)
    elif dense_mode == "sclip":
        patches = encode_sclip_tokens(model.visual, image)
    else:
        _, patches = encode_visual_tokens(model.visual, image)
    patches = F.normalize(patches.float(), dim=-1)

    logits = patches @ text_features.t()
    logits = logits / temperature
    grid = int(logits.shape[1] ** 0.5)
    logits = logits.reshape(1, grid, grid, len(VOC_CLASSES)).permute(0, 3, 1, 2)
    logits = F.interpolate(logits, size=(image.shape[-2], image.shape[-1]), mode="bilinear", align_corners=False)
    return logits.argmax(dim=1).squeeze(0).cpu() + 1


def update_confusion(confusion: torch.Tensor, pred: torch.Tensor, target: torch.Tensor) -> None:
    valid = (target >= 1) & (target <= len(VOC_CLASSES))
    pred = pred[valid] - 1
    target = target[valid] - 1
    if target.numel() == 0:
        return
    index = target * len(VOC_CLASSES) + pred.clamp(0, len(VOC_CLASSES) - 1)
    bins = torch.bincount(index, minlength=len(VOC_CLASSES) ** 2)
    confusion += bins.reshape(len(VOC_CLASSES), len(VOC_CLASSES))


def compute_metrics(confusion: torch.Tensor) -> dict[str, object]:
    confusion = confusion.float()
    tp = confusion.diag()
    row_sum = confusion.sum(dim=1)
    col_sum = confusion.sum(dim=0)
    union = row_sum + col_sum - tp
    iou = tp / union.clamp_min(1)
    acc = tp.sum() / confusion.sum().clamp_min(1)
    mean_acc = (tp / row_sum.clamp_min(1)).mean()
    present = row_sum > 0

    class_iou = {VOC_CLASSES[i]: float(iou[i]) for i in range(len(VOC_CLASSES))}
    return {
        "foreground_miou": float(iou[present].mean()),
        "foreground_pixel_acc": float(acc),
        "mean_class_acc": float(mean_acc),
        "class_iou": class_iou,
    }


def palette_mask(mask: torch.Tensor) -> Image.Image:
    colors = [
        (128, 0, 0),
        (0, 128, 0),
        (128, 128, 0),
        (0, 0, 128),
        (128, 0, 128),
        (0, 128, 128),
        (128, 128, 128),
        (64, 0, 0),
        (192, 0, 0),
        (64, 128, 0),
        (192, 128, 0),
        (64, 0, 128),
        (192, 0, 128),
        (64, 128, 128),
        (192, 128, 128),
        (0, 64, 0),
        (128, 64, 0),
        (0, 192, 0),
        (128, 192, 0),
        (0, 64, 128),
    ]
    mask = mask.clone()
    mask[(mask < 1) | (mask > len(VOC_CLASSES))] = 0
    rgb = torch.zeros(mask.shape[0], mask.shape[1], 3, dtype=torch.uint8)
    for index, color in enumerate(colors, start=1):
        rgb[mask == index] = torch.tensor(color, dtype=torch.uint8)
    return Image.fromarray(rgb.numpy(), mode="RGB")


def save_visualization(output_dir: Path, image_id: str, display: Image.Image, pred: torch.Tensor, target: torch.Tensor) -> None:
    pred_img = palette_mask(pred)
    target_img = palette_mask(target)
    overlay = Image.blend(display, pred_img, alpha=0.45)

    gap = 8
    width = display.width * 4 + gap * 3
    height = display.height
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in [display, target_img, pred_img, overlay]:
        canvas.paste(panel, (x, 0))
        x += display.width + gap
    canvas.save(output_dir / f"{image_id}_voc_seg.png")


def save_outputs(output_dir: Path, model_name: str, metrics: dict[str, object], confusion: torch.Tensor) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump({"model": model_name, **metrics}, file, ensure_ascii=False, indent=2)

    with open(output_dir / "class_iou.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "iou"])
        for class_name, iou in metrics["class_iou"].items():
            writer.writerow([class_name, iou])

    torch.save(confusion, output_dir / "confusion.pt")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    voc_root = Path(args.voc_root)
    paths = voc_paths(voc_root, args.split)
    if args.limit is not None:
        paths = paths[: args.limit]
    print(f"model={args.model_name} images={len(paths)} device={device} image_size={args.image_size}")
    print(f"dense_mode={args.dense_mode}")

    model = load_model(config, args.checkpoint, device)
    text_features = encode_text_features(model, device)

    confusion = torch.zeros(len(VOC_CLASSES), len(VOC_CLASSES), dtype=torch.long)
    for index, (image_path, mask_path, image_id) in enumerate(tqdm(paths, desc=args.model_name)):
        image, target, display = preprocess_pair(image_path, mask_path, args.image_size)
        pred = predict_mask(model, image, text_features, device, args.temperature, args.dense_mode)
        update_confusion(confusion, pred, target)
        if index < args.save_vis:
            save_visualization(vis_dir, image_id, display, pred, target)

    metrics = compute_metrics(confusion)
    save_outputs(output_dir, args.model_name, {"dense_mode": args.dense_mode, **metrics}, confusion)
    print(json.dumps({"model": args.model_name, "dense_mode": args.dense_mode, **metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
