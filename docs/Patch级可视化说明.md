# Patch 级对齐热力图说明

## 目的

ATAS 的核心目标不是单纯提升图像级分类，而是增强 CLIP image encoder 的局部 patch 表征，使其更适合 open-vocabulary dense prediction。因此除了 kNN 评估，还需要补充 patch-level 可视化。

本仓库新增 `scripts/visualize_patch_alignment.py`，用于可视化 patch token 与查询语义之间的相似度热力图。

## 默认可视化方式

默认查询为图像自身的 CLS token：

- OpenCLIP：展示原始 CLIP 中 patch 与全局 CLS 的相似度分布。
- ATAS：展示训练后 patch 与全局 CLS 的相似度分布。
- ATAS - OpenCLIP：展示训练后局部响应相对 baseline 的变化。

这种方式不依赖 ImageNet 类名映射，适合直接展示 ATAS 的 GLD 思路：让局部 patch 更好地吸收全局语义。

## 已完成 smoke test

已经用 CPU 跑通 2 张图的小规模测试：

```text
outputs/patch_alignment_vis_cpu_smoke/
```

示例输出：

- `patch_alignment_00_n02804610.png`
- `patch_alignment_01_n02480495.png`

## GPU 可视化结果

已经生成 epoch 3 的 8 张 448 分辨率 patch alignment 图：

```text
outputs/patch_alignment_vis_full_448/
```

继续训练到 epoch 6 后，也已生成对应的 8 张 448 分辨率可视化图：

```text
outputs/patch_alignment_vis_epoch6_448/
```

其中每张图包含四列：

1. 原始输入图像。
2. OpenCLIP patch-to-CLS 热力图。
3. ATAS patch-to-CLS 热力图。
4. ATAS 相对 OpenCLIP 的响应变化。

## 复现实验命令

使用如下命令生成 epoch 6 高清版本：

```bash
cd /mnt/t1b6/xuzhejia/atas-repro
PYTHONPATH=. /mnt/t1b6/xuzhejia/app/miniconda3/envs/atas/bin/python \
  scripts/visualize_patch_alignment.py \
  --config configs/atas_vitb_subset_100x200_stable.yaml \
  --data-root /mnt/t1b6/xuzhejia/datasets/imagenet_subset_100x200/train \
  --checkpoint outputs/atas_vitb_subset_100x200_stable/checkpoint_epoch_6.pt \
  --output-dir outputs/patch_alignment_vis_epoch6_448 \
  --device cpu \
  --num-images 8 \
  --image-size 448
```

说明：可视化脚本需要同时加载 OpenCLIP baseline 和 ATAS checkpoint。服务器 GPU 被其他任务占用时，用 CPU 跑 8 张图更稳。

## 汇报中如何解释

可以这样说：

> 图像级 kNN 说明 ATAS 没有破坏 CLIP 原有语义能力；patch-level 热力图进一步展示训练后局部 patch 对全局语义的响应发生变化，因此更贴近 dense prediction 任务需求。
