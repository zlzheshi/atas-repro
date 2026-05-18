# ATAS epoch 6 继续训练与评估结果

## 实验目的

在 epoch 3 已经得到稳定 checkpoint 的基础上，进一步把主实验训练到 6 个 epoch，观察更长训练是否能继续提升图像级表征，并检查这种提升是否会同步迁移到 VOC2012 零样本 patch-text 分割评估。

本实验使用的配置仍然是：

```text
configs/atas_vitb_subset_100x200_stable.yaml
```

数据集仍为 ImageNet-100x200 子集，共 100 类、每类 200 张图像。

## 继续训练设置

从 epoch 3 checkpoint 恢复训练：

```bash
cd /mnt/t1b6/xuzhejia/atas-repro
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. /mnt/t1b6/xuzhejia/app/miniconda3/envs/atas/bin/python train_atas.py \
  --config configs/atas_vitb_subset_100x200_stable.yaml \
  --data-root /mnt/t1b6/xuzhejia/datasets/imagenet_subset_100x200/train \
  --resume outputs/atas_vitb_subset_100x200_stable/checkpoint_epoch_3.pt
```

训练已完成到 epoch 6，当前主要 checkpoint 为：

```text
outputs/atas_vitb_subset_100x200_stable/checkpoint_epoch_6.pt
```

epoch 6 末尾日志中的关键 loss 为：

| 阶段 | loss | GLD | LLD | GGD | 检查项 |
| --- | ---: | ---: | ---: | ---: | --- |
| epoch 3 结束 | 6.1703 | 3.2133 | 0.4432 | 2.9526 | `grid=6, unique=36/36` |
| epoch 6 结束 | 6.1139 | 3.1421 | 0.4570 | 2.9672 | `grid=6, unique=36/36` |

可以看到继续训练后总 loss 和 GLD 仍有下降，说明局部 patch 到全局语义的蒸馏信号仍在继续学习；GGD 基本保持稳定，说明全局语义约束没有明显失控。

## ImageNet 子集 kNN 评估

评估设置：

- gallery：每类 160 张。
- query：每类 40 张。
- 指标：Top-1 kNN、Top-5 neighbor hit、centroid margin。

| 模型 | Top-1 kNN | Top-5 Hit | Centroid Margin |
| --- | ---: | ---: | ---: |
| OpenCLIP baseline | 0.8900 | 0.9653 | 0.0757 |
| ATAS epoch 3 | 0.8913 | 0.9618 | 0.2239 |
| ATAS epoch 6 | 0.8965 | 0.9650 | 0.2252 |

结论：

- epoch 6 的 Top-1 kNN 从 epoch 3 的 0.8913 提升到 0.8965，也超过原始 OpenCLIP baseline。
- centroid margin 保持在 0.2252，明显高于 baseline 的 0.0757。
- 这说明继续训练让 ImageNet 子集上的图像级语义检索更好，同时保持了 ATAS 训练后更强的类间分离趋势。

输出目录：

```text
outputs/eval_subset_100x200_epoch6_knn/
```

## VOC2012 零样本分割评估

评估设置：

- 数据集：PASCAL VOC2012 val split。
- 图像数：1449。
- 类别：20 个前景类别，不计背景类。
- 方法：ViT patch token 与 VOC 类别文本特征直接匹配，再上采样到像素级。

| 模型 | 前景 mIoU | 前景像素准确率 | 平均类别准确率 |
| --- | ---: | ---: | ---: |
| OpenCLIP baseline | 0.4016 | 0.5565 | 0.5680 |
| ATAS epoch 3 | 0.3604 | 0.5174 | 0.5789 |
| ATAS epoch 6 | 0.3287 | 0.4854 | 0.5462 |

结论：

- epoch 6 在 ImageNet kNN 上更好，但在 VOC2012 简化 patch-text 分割评估上低于 epoch 3 和原始 OpenCLIP。
- 这说明当前课程规模训练得到的特征更适合 ImageNet 子集的图像级检索，但没有直接转化为更好的 VOC patch-text segmentation。
- 该现象适合在报告中作为局限性分析：ATAS 的完整 dense prediction 效果依赖更大训练规模、更完整的下游分割框架和更细致的文本/背景建模；当前实现已经复现核心训练机制，但还不是论文完整 benchmark 级别。

输出目录：

```text
outputs/voc_zero_shot_seg_full_atas_epoch6/
```

## Patch 级可视化

已生成 epoch 6 的 448 分辨率 patch alignment 图，共 8 张。每张图包含原图、OpenCLIP 热力图、ATAS 热力图和差异热力图。

输出目录：

```text
outputs/patch_alignment_vis_epoch6_448/
```

## 汇报建议

最终汇报时建议这样组织：

1. 主结果使用 ImageNet 子集 kNN：epoch 6 的 Top-1 kNN 和 centroid margin 都优于 baseline，是最稳的正向定量证据。
2. 消融实验继续使用 epoch 3 阶段结果：它已经清楚说明 `GGD` 对保持全局语义很关键。
3. VOC2012 作为补充实验和局限性分析：诚实说明简化 patch-text 分割没有超过 baseline，体现复现工作的边界。
4. patch 可视化作为定性结果：展示 ATAS 训练确实改变了局部 patch 响应。

