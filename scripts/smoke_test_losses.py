import torch

from src.atas.losses import ATASLossConfig, atas_loss, atas_region_loss


def main() -> None:
    batch_size = 4
    num_patches = 16
    dim = 32

    student_cls = torch.randn(batch_size, dim, requires_grad=True)
    student_patches = torch.randn(batch_size, num_patches, dim, requires_grad=True)
    teacher_cls = torch.randn(batch_size, dim)
    teacher_patches = torch.randn(batch_size, num_patches, dim)

    loss, metrics = atas_loss(
        student_cls=student_cls,
        student_patches=student_patches,
        teacher_cls=teacher_cls,
        teacher_patches=teacher_patches,
        config=ATASLossConfig(),
    )
    loss.backward()

    region_boxes = torch.tensor(
        [
            [0, 0, 2, 0, 2],
            [0, 0, 2, 2, 4],
            [0, 2, 4, 0, 2],
            [0, 2, 4, 2, 4],
        ],
        dtype=torch.long,
    )
    region_loss, region_metrics = atas_region_loss(
        student_global_cls=student_cls,
        student_mosaic_patches=student_patches[:1],
        teacher_global_cls=teacher_cls,
        teacher_region_cls=teacher_cls,
        teacher_mosaic_patches=teacher_patches[:1],
        region_boxes=region_boxes,
        patch_grid=(4, 4),
        config=ATASLossConfig(),
    )
    region_loss.backward()

    print("atas_loss", {key: round(value.item(), 4) for key, value in metrics.items()})
    print("atas_region_loss", {key: round(value.item(), 4) for key, value in region_metrics.items()})


if __name__ == "__main__":
    main()
