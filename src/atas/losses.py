from __future__ import annotations

"""ATAS 三项蒸馏损失。

本文件把论文中的三个核心模块拆成独立函数：

- GLD（Global-to-Local Distillation）：用 teacher 的全局 CLS token 监督
  student 的局部 patch token，使局部 token 获得开放词汇语义。
- LLD（Local-to-Local Distillation）：约束 student/teacher patch token 之间的
  两两相似度结构，避免局部空间关系被训练破坏。
- GGD（Global-to-Global Distillation）：约束 student CLS token 继续对齐
  teacher CLS token，保留原始 CLIP 的全局图文语义。

训练使用 DDP 时，GLD/GGD 的 InfoNCE 负样本可以通过 all_gather 扩展到
全局 batch。这样 2 卡 batch=72 的有效负样本数就是 144，而不是每个 rank
各自只看到本地 72 个 teacher feature。
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ATASLossConfig:
    """ATAS 损失超参数。

    lambda_* 对应论文中三项损失的加权系数。`gather_distributed_negatives`
    是本复现额外显式暴露的工程选项，用于控制 DDP 下 InfoNCE 是否使用
    跨 GPU teacher features 作为负样本。
    """

    temperature: float = 1.0
    lambda_gld: float = 1.0
    lambda_lld: float = 0.01
    lambda_ggd: float = 1.0
    gather_distributed_negatives: bool = False


def normalize_features(x: Tensor) -> Tensor:
    """L2 normalize CLIP features before cosine / InfoNCE computation."""
    return F.normalize(x, dim=-1)


def distributed_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def gather_teacher_features(teacher: Tensor) -> tuple[Tensor, int]:
    """Gather teacher features across DDP ranks for global InfoNCE negatives.

    Args:
        teacher: 当前 rank 的 teacher feature，形状为 `[local_batch, dim]`。

    Returns:
        `(global_teacher, rank)`，其中 `global_teacher` 形状为
        `[world_size * local_batch, dim]`。student query 仍只来自本 rank，
        但 logits 的列扩展为所有 rank 的 teacher features。

    注意这里不需要对 teacher 反传梯度；teacher 本身是冻结模型，all_gather
    只用于构造更大的负样本集合。
    """
    if distributed_world_size() == 1:
        return teacher, 0

    teacher = teacher.contiguous()
    gathered = [torch.empty_like(teacher) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, teacher)
    return torch.cat(gathered, dim=0), dist.get_rank()


def contrastive_self_distill(
    student: Tensor,
    teacher: Tensor,
    temperature: float,
    gather_distributed_negatives: bool = False,
) -> Tensor:
    """One-way InfoNCE from student queries to teacher keys.

    `student[i]` 的正样本是同一图像或同一 mosaic cell 对应的 `teacher[i]`。
    其他 teacher features 都作为负样本。DDP all-gather 开启时，正样本在
    拼接后的 teacher 矩阵中不再位于 `i`，而是位于
    `rank * local_batch + i`，因此需要 `target_offset` 修正标签。
    """
    student = normalize_features(student)
    teacher = normalize_features(teacher)

    target_offset = 0
    if gather_distributed_negatives:
        local_batch = student.shape[0]
        teacher, rank = gather_teacher_features(teacher)
        target_offset = rank * local_batch

    logits = student @ teacher.t()
    logits = logits / temperature
    targets = torch.arange(logits.shape[0], device=logits.device) + target_offset
    return F.cross_entropy(logits, targets)


def global_to_local_loss(
    student_patches: Tensor,
    teacher_cls: Tensor,
    temperature: float,
    gather_distributed_negatives: bool = False,
) -> Tensor:
    """ATAS GLD loss.

    Args:
        student_patches: Student patch features with shape [batch, num_patches, dim].
        teacher_cls: Teacher CLS features with shape [batch, dim].
        temperature: Contrastive temperature.
    """
    student_patches = normalize_features(student_patches)
    teacher_cls = normalize_features(teacher_cls)

    # 先计算每个 patch 与 teacher CLS 的相似度，再做 softmax 加权池化。
    # 这样 GLD 不是平均所有 patch，而是更关注与图像语义最相关的局部区域。
    patch_weights = torch.einsum("bnd,bd->bn", student_patches, teacher_cls)
    patch_weights = F.softmax(patch_weights / temperature, dim=1)
    aggregated_local = torch.einsum("bn,bnd->bd", patch_weights, student_patches)

    return contrastive_self_distill(
        aggregated_local,
        teacher_cls,
        temperature,
        gather_distributed_negatives=gather_distributed_negatives,
    )


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
        # region_boxes 使用 patch-grid 坐标，而不是像素坐标。
        # 例如 960x960 输入、ViT-B/16 patch size=16 时，patch grid 是 60x60；
        # 6x6 mosaic 的每个 cell 对应 10x10 个 patch。
        region = patch_map[mosaic_index, row_start:row_end, col_start:col_end].reshape(-1, dim)
        region = normalize_features(region)
        # 对每个 source image 的区域单独做 teacher-guided pooling，得到一个
        # `[dim]` 区域向量，再和该 source image 的 teacher CLS 做 InfoNCE。
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
    gather_distributed_negatives: bool = False,
) -> Tensor:
    """GLD loss for mosaic training, aligning each cell with its source CLS token.

    标准 `global_to_local_loss` 默认一张图对应一组 patch。mosaic 训练中，一张
    960x960 图由多个 source images 拼成，因此这里先按 cell 区域池化，再让
    每个 cell 的 student local feature 对齐它自己的 teacher CLS。
    """
    pooled_regions = weighted_region_pool(
        student_patches=student_patches,
        teacher_cls=teacher_cls,
        region_boxes=region_boxes,
        patch_grid=patch_grid,
        temperature=temperature,
    )
    return contrastive_self_distill(
        pooled_regions,
        teacher_cls,
        temperature,
        gather_distributed_negatives=gather_distributed_negatives,
    )


def local_to_local_loss(student_patches: Tensor, teacher_patches: Tensor) -> Tensor:
    """ATAS LLD loss preserving pairwise patch-similarity structure.

    LLD 不直接要求 student patch 等于 teacher patch，而是要求 patch-patch
    相似度矩阵一致。这样它约束的是局部结构关系：哪些 patch 彼此相似、
    哪些 patch 应该分开。该项在 960x960 mosaic 下矩阵很大，因此训练入口
    支持 `max_lld_patches` 随机采样 patch，降低显存占用。
    """
    student_patches = normalize_features(student_patches)
    teacher_patches = normalize_features(teacher_patches)

    # [B, N, D] x [B, D, N] -> [B, N, N]，每个 batch 内独立计算 patch 关系。
    student_rel = torch.bmm(student_patches, student_patches.transpose(1, 2))
    teacher_rel = torch.bmm(teacher_patches, teacher_patches.transpose(1, 2))
    return F.mse_loss(student_rel, teacher_rel)


def global_to_global_loss(
    student_cls: Tensor,
    teacher_cls: Tensor,
    temperature: float,
    gather_distributed_negatives: bool = False,
) -> Tensor:
    """ATAS GGD loss preserving the teacher's global CLIP semantics.

    GLD 会推动局部 token 获得全局语义，但如果只优化 GLD/LLD，student 的
    CLS token 可能偏离原始 CLIP。GGD 用同样的 InfoNCE 形式保持全局图像
    表征与冻结 teacher 对齐。
    """
    return contrastive_self_distill(
        student_cls,
        teacher_cls,
        temperature,
        gather_distributed_negatives=gather_distributed_negatives,
    )


def atas_loss(
    student_cls: Tensor,
    student_patches: Tensor,
    teacher_cls: Tensor,
    teacher_patches: Tensor,
    config: ATASLossConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the weighted ATAS objective for non-mosaic token pairs."""
    loss_gld = global_to_local_loss(
        student_patches,
        teacher_cls,
        config.temperature,
        gather_distributed_negatives=config.gather_distributed_negatives,
    )
    loss_lld = local_to_local_loss(student_patches, teacher_patches)
    loss_ggd = global_to_global_loss(
        student_cls,
        teacher_cls,
        config.temperature,
        gather_distributed_negatives=config.gather_distributed_negatives,
    )

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
    """Compute ATAS when global tokens come from individual images and local tokens from mosaics.

    训练批次中同时包含：

    - `global_images`：原始单图，送入 teacher/student 得到 CLS token；
    - `mosaic_images`：由多张图拼成的大图，送入 teacher/student 得到 dense patch token；
    - `region_boxes`：记录每个 source image 在 mosaic patch grid 中的区域。

    因此 GLD 使用 `teacher_region_cls` 对齐每个 mosaic cell，LLD 使用整张
    mosaic 的 patch token 约束局部结构，GGD 使用原始单图 CLS 约束全局语义。
    """
    loss_gld = region_global_to_local_loss(
        student_patches=student_mosaic_patches,
        teacher_cls=teacher_region_cls,
        region_boxes=region_boxes,
        patch_grid=patch_grid,
        temperature=config.temperature,
        gather_distributed_negatives=config.gather_distributed_negatives,
    )

    if max_lld_patches is not None and student_mosaic_patches.shape[1] > max_lld_patches:
        # 960x960 / 16 = 60，因此完整 patch 关系矩阵是 3600x3600。
        # 直接算完整 LLD 显存代价很高；随机采样 1024 个 patch 是本复现采用的
        # 工程近似，文档中也将它列为与作者实现可能存在差异的点。
        indices = torch.randperm(student_mosaic_patches.shape[1], device=student_mosaic_patches.device)
        indices = indices[:max_lld_patches]
        student_lld_patches = student_mosaic_patches[:, indices]
        teacher_lld_patches = teacher_mosaic_patches[:, indices]
    else:
        student_lld_patches = student_mosaic_patches
        teacher_lld_patches = teacher_mosaic_patches

    loss_lld = local_to_local_loss(student_lld_patches, teacher_lld_patches)
    loss_ggd = global_to_global_loss(
        student_global_cls,
        teacher_global_cls,
        config.temperature,
        gather_distributed_negatives=config.gather_distributed_negatives,
    )

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
