from __future__ import annotations

import random
from collections.abc import Sequence

from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def build_mosaic(images: Sequence[Image.Image], grid_size: int, output_size: int) -> Image.Image:
    """Create a square mosaic from object-centric images."""
    expected = grid_size * grid_size
    if len(images) != expected:
        raise ValueError(f"grid_size={grid_size} requires {expected} images, got {len(images)}")

    cell_size = output_size // grid_size
    mosaic = Image.new("RGB", (output_size, output_size))

    for index, image in enumerate(images):
        row, col = divmod(index, grid_size)
        image = image.convert("RGB")
        image = TF.resize(image, [cell_size, cell_size], interpolation=InterpolationMode.BICUBIC)
        mosaic.paste(image, (col * cell_size, row * cell_size))

    return mosaic


def choose_mosaic_grid(grid_choices: Sequence[int]) -> int:
    return random.choice(list(grid_choices))

