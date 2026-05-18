# ATAS 子集 kNN 评估结果

## 评估目的

上一阶段已经验证 ATAS 训练 loss 能正常下降。本阶段进一步评估训练后的视觉特征是否保持语义分类能力，并观察特征空间是否发生有效变化。

## 评估设置

- 数据集：ImageNet 子集，100 类，每类 200 张，共 20,000 张
- 划分方式：每类 160 张作为 gallery，每类 40 张作为 query
- 评估方式：基于图像特征的 kNN 分类
- 对比对象：
  - `openclip_baseline`：原始 OpenCLIP ViT-B/16
  - `checkpoint_epoch_3`：ATAS 子集训练 3 epoch 后的 checkpoint
  - `checkpoint_epoch_6`：ATAS 子集继续训练到 6 epoch 后的 checkpoint
- 主要 checkpoint：`outputs/atas_vitb_subset_100x200_stable/checkpoint_epoch_6.pt`

## 结果表

| 模型 | Top-1 kNN | Top-5 Neighbor Hit | Own Centroid Sim | Nearest Other Sim | Centroid Margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenCLIP baseline | 0.8900 | 0.9653 | 0.8602 | 0.7845 | 0.0757 |
| ATAS epoch 3 | 0.8913 | 0.9618 | 0.6734 | 0.4495 | 0.2239 |
| ATAS epoch 6 | 0.8965 | 0.9650 | 0.6734 | 0.4482 | 0.2252 |

## 结果解读

1. ATAS epoch 3 后 Top-1 kNN 从 `0.8900` 到 `0.8913`，说明小规模自蒸馏没有破坏原始 CLIP 的语义判别能力。
2. 继续训练到 epoch 6 后，Top-1 kNN 进一步提升到 `0.8965`，Top-5 hit 基本回到 baseline 水平。
3. Centroid margin 从 baseline 的 `0.0757` 提升到 epoch 3 的 `0.2239`，epoch 6 继续保持在 `0.2252`，说明类间分离趋势稳定存在。
4. Baseline 与 epoch 6 query 特征的同图 cosine 均值为 `0.6641`，说明训练后的特征不是简单复制原始 CLIP，而是发生了实质性调整。

## 汇报可用结论

在 100 类 ImageNet 子集上，ATAS 训练到 epoch 6 后不但保持了 OpenCLIP 的语义分类能力，还把 Top-1 kNN 提升到 `0.8965`，同时维持明显更高的 centroid margin。该结果可以作为课程复现中的主要定量证据：训练流程不仅能运行，得到的 checkpoint 也在轻量评估中表现出可解释、可量化的特征变化。

## 复现实验命令

```bash
cd /mnt/t1b6/xuzhejia/atas-repro
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /mnt/t1b6/xuzhejia/app/miniconda3/envs/atas/bin/python \
  scripts/evaluate_imagenet_subset_knn.py \
  --config configs/atas_vitb_subset_100x200_stable.yaml \
  --data-root /mnt/t1b6/xuzhejia/datasets/imagenet_subset_100x200/train \
  --checkpoint outputs/atas_vitb_subset_100x200_stable/checkpoint_epoch_6.pt \
  --output-dir outputs/eval_subset_100x200_epoch6_knn \
  --batch-size 256 \
  --num-workers 8
```

评估输出：

- `outputs/eval_subset_100x200_epoch6_knn/metrics.json`
- `outputs/eval_subset_100x200_epoch6_knn/metrics.csv`
- `/mnt/t1b6/xuzhejia/logs/eval_subset_100x200_epoch6_knn.log`
