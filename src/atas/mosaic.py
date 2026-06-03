from __future__ import annotations

"""Mosaic image construction utilities used by ATAS training.

ATAS 需要在高分辨率输入中同时看到多个 source images，训练时将 `grid_size *
grid_size` 张图缩放后拼成一张 square mosaic。例如 `grid_size=6`、`output_size=960`
时，每个 cell 是 160x160 像素；对 ViT-B/16 来说，每个 cell 对应 10x10 个
patch token。训练入口会额外记录每个 cell 在 patch grid 中的位置，用于 GLD
把该 cell 的局部 token 对齐到对应 source image 的 teacher CLS token。
"""

import random
from collections.abc import Sequence

from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def build_mosaic(images: Sequence[Image.Image], grid_size: int, output_size: int) -> Image.Image:
    """Create a square mosaic from object-centric images.

    Args:
        images: 长度必须等于 `grid_size ** 2` 的 PIL 图像列表。
        grid_size: mosaic 每行/每列的 cell 数。
        output_size: 输出方形图像边长，单位为像素。

    Returns:
        RGB PIL image with shape `[output_size, output_size]`.
    """
    expected = grid_size * grid_size
    if len(images) != expected:
        raise ValueError(f"grid_size={grid_size} requires {expected} images, got {len(images)}")

    cell_size = output_size // grid_size
    mosaic = Image.new("RGB", (output_size, output_size))

    for index, image in enumerate(images):
        row, col = divmod(index, grid_size)
        image = image.convert("RGB")
        # 论文设置中 cell 需要精确落在 ViT patch grid 上，因此 output_size 和
        # grid_size 应选择能被 patch size=16 整除的组合，例如 960/6/16=10。
        image = TF.resize(image, [cell_size, cell_size], interpolation=InterpolationMode.BICUBIC)
        mosaic.paste(image, (col * cell_size, row * cell_size))

    return mosaic


def choose_mosaic_grid(grid_choices: Sequence[int]) -> int:
    """Randomly choose one mosaic grid size for the current batch."""
    return random.choice(list(grid_choices))
