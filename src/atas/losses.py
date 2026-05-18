from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ATASLossConfig:
    temperature: float = 1.0
    lambda_gld: float = 1.0
    lambda_lld: float = 0.01
    lambda_ggd: float = 1.0


def normalize_features(x: Tensor) -> Tensor:
    return F.normalize(x, dim=-1)


def contrastive_self_distill(student: Tensor, teacher: Tensor, temperature: float) -> Tensor:
    """One-way contrastive loss between matched student and teacher features."""
    student = normalize_features(student)
    teacher = normalize_features(teacher)
    logits = student @ teacher.t()
    logits = logits / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, targets)


def global_to_local_loss(student_patches: Tensor, teacher_cls: Tensor, temperature: float) -> Tensor:
    """ATAS GLD loss.

    Args:
        student_patches: Student patch features with shape [batch, num_patches, dim].
        teacher_cls: Teacher CLS features with shape [batch, dim].
        temperature: Contrastive temperature.
    """
    student_patches = normalize_features(student_patches)
    teacher_cls = normalize_features(teacher_cls)

    patch_weights = torch.einsum("bnd,bd->bn", student_patches, teacher_cls)
    patch_weights = F.softmax(patch_weights / temperature, dim=1)
    aggregated_local = torch.einsum("bn,bnd->bd", patch_weights, student_patches)

    return contrastive_self_distill(aggregated_local, teacher_cls, temperature)


def weighted_region_pool(
    student_patches: Tensor,
    teacher_cls: Tensor,
    region_boxes: Tensor,
    patch_grid: tuple[int, int],
    temperature: float,
) -> Tensor:
    """Pool mosaic patch features inside each source-image region.

    Args:
        student_patches: Patch features from mosaic images, [num_mosaics, num_patches, dim].
        teacher_cls: Teacher CLS features for the corresponding source images, [num_regions, dim].
        region_boxes: Integer tensor [num_regions, 5] with columns
            [mosaic_index, row_start, row_end, col_start, col_end] in patch-grid units.
        patch_grid: Spatial patch grid of the mosaic feature map.
        temperature: Softmax temperature for selective patch weighting.
    """
    grid_h, grid_w = patch_grid
    dim = student_patches.shape[-1]
    patch_map = student_patches.reshape(student_patches.shape[0], grid_h, grid_w, dim)
    teacher_cls = normalize_features(teacher_cls)

    pooled_regions: list[Tensor] = []
    for index, box in enumerate(region_boxes.tolist()):
        mosaic_index, row_start, row_end, col_start, col_end = box
        region = patch_map[mosaic_index, row_start:row_end, col_start:col_end].reshape(-1, dim)
        region = normalize_features(region)
        weights = region @ teacher_cls[index]
        weights = F.softmax(weights / temperature, dim=0)
        pooled_regions.append(torch.sum(weights[:, None] * region, dim=0))

    return torch.stack(pooled_regions, dim=0)


def region_global_to_local_loss(
    student_patches: Tensor,
    teacher_cls: Tensor,
    region_boxes: Tensor,
    patch_grid: tuple[int, int],
    temperature: float,
) -> Tensor:
    """GLD loss for mosaic training, aligning each cell with its source CLS token."""
    pooled_regions = weighted_region_pool(
        student_patches=student_patches,
        teacher_cls=teacher_cls,
        region_boxes=region_boxes,
        patch_grid=patch_grid,
        temperature=temperature,
    )
    return contrastive_self_distill(pooled_regions, teacher_cls, temperature)


def local_to_local_loss(student_patches: Tensor, teacher_patches: Tensor) -> Tensor:
    """ATAS LLD loss preserving pairwise patch-similarity structure."""
    student_patches = normalize_features(student_patches)
    teacher_patches = normalize_features(teacher_patches)

    student_rel = torch.bmm(student_patches, student_patches.transpose(1, 2))
    teacher_rel = torch.bmm(teacher_patches, teacher_patches.transpose(1, 2))
    return F.mse_loss(student_rel, teacher_rel)


def global_to_global_loss(student_cls: Tensor, teacher_cls: Tensor, temperature: float) -> Tensor:
    """ATAS GGD loss preserving the teacher's global CLIP semantics."""
    return contrastive_self_distill(student_cls, teacher_cls, temperature)


def atas_loss(
    student_cls: Tensor,
    student_patches: Tensor,
    teacher_cls: Tensor,
    teacher_patches: Tensor,
    config: ATASLossConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the weighted ATAS objective."""
    loss_gld = global_to_local_loss(student_patches, teacher_cls, config.temperature)
    loss_lld = local_to_local_loss(student_patches, teacher_patches)
    loss_ggd = global_to_global_loss(student_cls, teacher_cls, config.temperature)

    total = (
        config.lambda_gld * loss_gld
        + config.lambda_lld * loss_lld
        + config.lambda_ggd * loss_ggd
    )

    metrics = {
        "loss": total.detach(),
        "loss_gld": loss_gld.detach(),
        "loss_lld": loss_lld.detach(),
        "loss_ggd": loss_ggd.detach(),
    }
    return total, metrics


def atas_region_loss(
    student_global_cls: Tensor,
    student_mosaic_patches: Tensor,
    teacher_global_cls: Tensor,
    teacher_region_cls: Tensor,
    teacher_mosaic_patches: Tensor,
    region_boxes: Tensor,
    patch_grid: tuple[int, int],
    config: ATASLossConfig,
    max_lld_patches: int | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute ATAS when global tokens come from individual images and local tokens from mosaics."""
    loss_gld = region_global_to_local_loss(
        student_patches=student_mosaic_patches,
        teacher_cls=teacher_region_cls,
        region_boxes=region_boxes,
        patch_grid=patch_grid,
        temperature=config.temperature,
    )

    if max_lld_patches is not None and student_mosaic_patches.shape[1] > max_lld_patches:
        indices = torch.randperm(student_mosaic_patches.shape[1], device=student_mosaic_patches.device)
        indices = indices[:max_lld_patches]
        student_lld_patches = student_mosaic_patches[:, indices]
        teacher_lld_patches = teacher_mosaic_patches[:, indices]
    else:
        student_lld_patches = student_mosaic_patches
        teacher_lld_patches = teacher_mosaic_patches

    loss_lld = local_to_local_loss(student_lld_patches, teacher_lld_patches)
    loss_ggd = global_to_global_loss(student_global_cls, teacher_global_cls, config.temperature)

    total = (
        config.lambda_gld * loss_gld
        + config.lambda_lld * loss_lld
        + config.lambda_ggd * loss_ggd
    )

    metrics = {
        "loss": total.detach(),
        "loss_gld": loss_gld.detach(),
        "loss_lld": loss_lld.detach(),
        "loss_ggd": loss_ggd.detach(),
    }
    return total, metrics
