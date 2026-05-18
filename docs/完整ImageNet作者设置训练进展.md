# 完整 ImageNet 作者设置训练进展

## 目标

为了尽量接近 ATAS 论文设置，当前已经从 2 万张 ImageNet 子集训练切换到完整 ImageNet 训练。这个实验是目前最关键的一步，因为作者论文中的主要提升来自完整 ImageNet 规模的自蒸馏，而不是小子集训练。

## 对齐作者设置

当前配置文件：

```text
configs/atas_vitb_imagenet_full_author.yaml
```

核心设置：

| 项目 | 当前复现设置 | 作者论文设置 |
| --- | --- | --- |
| Backbone | OpenCLIP ViT-B/16 | CLIP ViT-B/16 |
| 训练数据 | 完整 ImageNet train | 完整 ImageNet train |
| 训练轮数 | 6 epochs | 6 epochs |
| GPU | 4 张 RTX A6000 | 4 张 RTX 3090 |
| batch size | 36 per GPU | 36 per GPU |
| 优化器 | AdamW | AdamW |
| 学习率 | 1e-5 | 1e-5 |
| weight decay | 0.1 | 0.1 |
| 损失权重 | GLD=1, LLD=0.01, GGD=1 | GLD=1, LLD=0.01, GGD=1 |
| temperature | 1.0 | 1.0 |
| mosaic | 6x6 | 论文训练使用 mosaic |

## 已完成检查

DDP smoke test 已完成：

- 使用完整 ImageNet 数据。
- 使用 `torch.distributed.run` 启动。
- 2 个 step 正常跑通。
- `grid=6`，`unique=36/36`，说明每个 mosaic 样本由 36 张不同图像组成。
- 单卡 smoke 时完整 ImageNet 每 epoch 约 35587 step；4 卡 DDP 正式训练每 epoch 约 8896 step。

## 当前训练状态

完整训练已经启动，日志文件：

```text
/mnt/t1b6/xuzhejia/logs/atas_full_author_wait.log
```

启动方式：

```bash
nohup bash scripts/run_full_imagenet_author_wait.sh > /mnt/t1b6/xuzhejia/logs/atas_full_author_wait.log 2>&1 &
```

当前状态记录：

- 等待脚本已在 4 张 GPU 空闲后自动启动训练。
- 训练使用 GPU 0,1,2,3。
- 正式训练正在跑 epoch 1。
- 日志中已看到 loss 从约 6.75 下降到约 6.1 左右，GLD/LLD/GGD 都在正常记录。
- 当前训练是目前最接近作者训练设置的复现实验。

训练输出目录：

```text
outputs/atas_vitb_imagenet_full_author/
```

checkpoint 预计按 epoch 保存：

```text
outputs/atas_vitb_imagenet_full_author/checkpoint_epoch_1.pt
outputs/atas_vitb_imagenet_full_author/checkpoint_epoch_2.pt
...
outputs/atas_vitb_imagenet_full_author/checkpoint_epoch_6.pt
```

## checkpoint 产生后的评估顺序

每个完整 ImageNet checkpoint 产生后，优先补以下评估：

1. VOC2012 vanilla zero-shot segmentation。
2. ImageNet 子集 kNN，用来检查全局语义是否保持。
3. patch alignment 可视化，用来观察局部响应变化。
4. 更可靠的 MaskCLIP/SCLIP 评估接入。

短期最关键指标是 VOC2012 vanilla mIoU。作者论文中 Vanilla CLIP baseline 为 41.8 mIoU，ATAS 为 56.0 mIoU；我们当前 OpenCLIP baseline 为 40.16 mIoU，具有可比性。完整 ImageNet checkpoint 是否能显著超过 40.16，是判断复现是否向论文主结果靠近的核心。
