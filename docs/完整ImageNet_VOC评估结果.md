# 完整 ImageNet 训练后的 VOC2012 评估结果

## 最新 QuickGELU 作者参数对齐实验

旧实验发现 OpenAI CLIP ViT-B/16 权重需要显式启用 QuickGELU。修正后，我们补充了一轮作者参数对齐训练：

```text
configs/atas_vitb_imagenet_full_author_quickgelu_2gpu_accum2.yaml
outputs/atas_vitb_imagenet_full_author_quickgelu_2gpu_accum2/checkpoint_epoch_6.pt
```

训练设置：

- ImageNet 全量训练。
- OpenAI CLIP ViT-B/16 + QuickGELU。
- `960x960` 输入，`6x6` mosaic。
- 6 epochs。
- 每卡 batch size 36，2 GPU，gradient accumulation 2，等效优化 batch 144。
- AdamW，`lr=1e-5`，`weight_decay=0.1`。
- `GLD=1.0, LLD=0.01, GGD=1.0`。

评估结果：

| 设置 | 模型 | 前景 mIoU | 前景像素准确率 | 平均类别准确率 |
| --- | --- | ---: | ---: | ---: |
| Vanilla | QuickGELU CLIP baseline | 0.3604 | 0.5191 | 0.5111 |
| Vanilla | QuickGELU ATAS epoch 6 | 0.3111 | 0.4729 | 0.5240 |
| SCLIP 风格 | QuickGELU CLIP baseline | 0.7650 | 0.8602 | 0.8790 |
| SCLIP 风格 | QuickGELU ATAS epoch 6 | 0.5703 | 0.7024 | 0.7750 |

表征漂移诊断：

| 模型 | CLS cosine | Global patch cosine | Mosaic patch cosine | Mosaic patch pairwise MSE |
| --- | ---: | ---: | ---: | ---: |
| QuickGELU baseline | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| QuickGELU ATAS epoch 6 | 0.6816 | 0.5106 | -0.0393 | 0.5448 |

结论：

- QuickGELU 修正后，SCLIP baseline 已经接近论文中 SCLIP CLIP 的量级。
- 但 ATAS checkpoint 仍显著低于 baseline，尤其 SCLIP 风格评估下降明显。
- 当前最重要的技术证据是 patch token 漂移：mosaic patch 与 teacher 的平均 cosine 已接近 0，说明局部表征结构被破坏。
- 详细代码审计和后续优先级见 [作者参数 QuickGELU 复现实验审计](作者参数QuickGELU复现实验审计.md)。

## All-Gather 与 Semantic Guard Probe

为排查多卡训练中 InfoNCE 负样本不足的问题，代码新增了：

```yaml
training:
  gather_distributed_negatives: true
```

开启后，GLD/GGD 在 DDP 进程间收集 teacher features。这样每张卡上的 student query 不再只和本地 teacher batch 对比，而是和全局 teacher batch 对比。

### Probe 配置

| 配置 | 说明 |
| --- | --- |
| `configs/atas_vitb_subset_100x200_quickgelu_allgather_probe.yaml` | QuickGELU，作者损失权重，80 steps，2GPU all-gather |
| `configs/atas_vitb_subset_100x200_quickgelu_semantic_guard_allgather_probe.yaml` | QuickGELU，保守语义约束，160 steps，2GPU all-gather |

### Probe 结果

| 设置 | Vanilla mIoU | SCLIP mIoU | kNN Top-1 | Mosaic patch cosine | Patch pairwise MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| QuickGELU all-gather probe | 0.3615 | 0.6901 | 0.9083 | 0.0595 | 0.3762 |
| QuickGELU semantic guard + all-gather probe | 0.3522 | 0.7410 | 0.9113 | 0.4667 | 0.0257 |

结论：

- 在子集 probe 中单独加入 all-gather 后，SCLIP mIoU 从完整作者参数 QuickGELU ATAS 的 `0.5703` 回升到 `0.6901`。
- semantic guard 与 all-gather 组合后，SCLIP mIoU 进一步达到 `0.7410`，已经接近 QuickGELU SCLIP baseline 的 `0.7650`。
- 该组合同时把 mosaic patch pairwise MSE 降到 `0.0257`，说明 patch 结构被明显保留下来。
- 这支持当前判断：作者参数下局部迁移过强导致 patch token 漂移，是复现失败的关键原因之一；全局 batch 负样本有帮助，但单独补齐并不足以解决完整训练的漂移。

### 完整 ImageNet b72 all-gather 结果

为进一步对齐作者的全局 batch，我们完成了完整 ImageNet 训练：

```text
configs/atas_vitb_imagenet_full_author_quickgelu_2gpu_b72_allgather.yaml
```

该配置使用：

- 2 GPU。
- 每卡 batch size 72。
- `gradient_accumulation_steps=1`。
- `gather_distributed_negatives=true`。
- 单步全局负样本数为 `72 x 2 = 144`。
- 等效优化 batch 也是 `144`。

相比此前的 2GPU batch36 accum2 版本，该配置同时对齐了优化 batch 和 InfoNCE 负样本数。

评估结果：

| Dense mode | Foreground mIoU | Pixel Acc | Mean Class Acc |
| --- | ---: | ---: | ---: |
| Vanilla | 0.3090 | 0.4699 | 0.5198 |
| SCLIP 风格 | 0.5817 | 0.7011 | 0.7803 |

表征漂移诊断：

| 模型 | CLS cosine | Global patch cosine | Mosaic patch cosine | Mosaic patch pairwise MSE |
| --- | ---: | ---: | ---: | ---: |
| QuickGELU baseline | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| QuickGELU b72 all-gather epoch 6 | 0.6876 | 0.5388 | -0.0513 | 0.4838 |

结论：

- b72 all-gather 完整训练的 SCLIP mIoU 为 `0.5817`，只比此前 batch36 accum2 的 `0.5703` 略高，仍明显低于 QuickGELU baseline `0.7650`。
- all-gather 补齐了全局负样本，但没有解决作者参数下的 patch token 漂移。
- 下一步不应继续原样加长作者参数训练；更合理的是把子集上有效的 semantic guard 或更强 LLD/学习率调度扩展到完整 ImageNet，并在报告中明确这属于改进复现而非严格作者参数复现。

## 实验背景

完整 ImageNet 作者设置训练已经完成，最终 checkpoint 为：

```text
outputs/atas_vitb_imagenet_full_author/checkpoint_epoch_6.pt
```

训练设置与作者论文主训练设置基本对齐：完整 ImageNet、6 epochs、4 卡 DDP、batch size 36 per GPU、AdamW、学习率 1e-5、weight decay 0.1，损失权重为 `GLD=1, LLD=0.01, GGD=1`。

训练完成后，我们补充了两组 VOC2012 zero-shot dense prediction 评估：

1. vanilla patch matching：直接用 ViT patch token 与 VOC 类别文本特征匹配。
2. SCLIP 风格评估：在最后一层 ViT block 使用 self-correlation attention 得到 dense patch 表征。

## Vanilla VOC2012 评估

运行脚本：

```bash
GPU=0 bash scripts/run_voc_full_author_sweep.sh
```

结果目录：

```text
outputs/voc_full_author_sweep/
```

| 模型 | 前景 mIoU | 前景像素准确率 | 平均类别准确率 |
| --- | ---: | ---: | ---: |
| OpenCLIP baseline | 0.4016 | 0.5565 | 0.5680 |
| ATAS epoch 1 | 0.3187 | 0.4770 | 0.5364 |
| ATAS epoch 2 | 0.3108 | 0.4709 | 0.5238 |
| ATAS epoch 6 | 0.3029 | 0.4672 | 0.5152 |
| ATAS epoch 3 | 0.3003 | 0.4616 | 0.5122 |
| ATAS epoch 4 | 0.2981 | 0.4625 | 0.5114 |
| ATAS epoch 5 | 0.2925 | 0.4567 | 0.5018 |

结论：

- 在当前 vanilla patch matching 评估中，完整 ImageNet ATAS checkpoint 仍未超过 OpenCLIP baseline。
- 最好的 ATAS checkpoint 是 epoch 1，前景 mIoU 为 0.3187；最终 epoch 6 为 0.3029。
- 这说明“直接 patch token 与文本特征匹配”的评估方式没有复现论文中 Vanilla ATAS 的提升。

## SCLIP 风格 VOC2012 评估

运行脚本：

```bash
GPU=0 bash scripts/run_voc_sclip_eval.sh
```

结果目录：

```text
outputs/voc_sclip_full_author/
```

| 模型 | 前景 mIoU | 前景像素准确率 | 平均类别准确率 |
| --- | ---: | ---: | ---: |
| OpenCLIP baseline + SCLIP | 0.6826 | 0.8034 | 0.8292 |
| ATAS epoch 1 + SCLIP | 0.4410 | 0.5731 | 0.6435 |
| ATAS epoch 6 + SCLIP | 0.4244 | 0.5618 | 0.6388 |
| ATAS epoch 3 + SCLIP | 0.4001 | 0.5310 | 0.5989 |

结论：

- SCLIP 风格推理显著提升了 OpenCLIP baseline，说明 dense inference 框架本身对 VOC2012 很关键。
- 但 ATAS checkpoint 在 SCLIP 风格评估中仍低于 OpenCLIP baseline。
- 这意味着当前复现中的 ATAS 训练并没有把 patch token 调整到有利于 VOC2012 zero-shot segmentation 的方向，至少没有在我们实现的 vanilla/SCLIP 评估下体现出作者论文中的提升。

## Semantic Guard Probe

为排查 ATAS checkpoint 下游下降的原因，我们补充了一组保护语义的子集 probe：

```text
configs/atas_vitb_subset_100x200_semantic_guard_probe.yaml
```

该配置相对作者设置做了保守调整：

- 学习率从 `1e-5` 降到 `5e-6`。
- weight decay 从 `0.1` 降到 `0.05`。
- `GLD` 权重从 `1.0` 降到 `0.25`。
- `GGD` 权重从 `1.0` 提到 `4.0`。
- 在 ImageNet-100x200 子集上训练 160 steps。

### 表征漂移诊断

诊断脚本：

```bash
python scripts/diagnose_checkpoint_drift.py \
  --config configs/atas_vitb_subset_100x200_stable.yaml \
  --data-root /mnt/t1b6/xuzhejia/datasets/imagenet_subset_100x200/train \
  --checkpoint full_epoch1=outputs/atas_vitb_imagenet_full_author/checkpoint_epoch_1.pt \
  --checkpoint full_epoch6=outputs/atas_vitb_imagenet_full_author/checkpoint_epoch_6.pt \
  --checkpoint semantic_guard_probe=outputs/atas_vitb_subset_100x200_semantic_guard_probe/checkpoint_epoch_1.pt \
  --output-dir outputs/checkpoint_drift_full_author_subset_after_probe
```

关键结果：

| 模型 | CLS cosine | Global patch cosine | Mosaic patch cosine | Patch pairwise MSE |
| --- | ---: | ---: | ---: | ---: |
| OpenCLIP baseline | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Full ATAS epoch 1 | 0.6627 | 0.6610 | -0.0367 | 0.5659 |
| Full ATAS epoch 6 | 0.6484 | 0.5159 | -0.0430 | 0.5764 |
| Semantic guard probe | 0.6905 | 0.9555 | 0.6633 | 0.0102 |

结论：

- Full ATAS checkpoint 的 mosaic patch token 与 teacher 几乎失去正相关，说明训练明显改变了局部表征结构。
- Semantic guard probe 显著降低 patch 表征漂移，尤其是 patch pairwise MSE 从约 `0.57` 降到 `0.0102`。
- 这支持一个判断：当前复现效果下降，很可能与 ATAS 训练中过强的局部迁移/局部表征漂移有关。

### 子集 kNN

| 模型 | Top-1 kNN | Top-5 | 类中心间隔 |
| --- | ---: | ---: | ---: |
| OpenCLIP baseline | 0.8900 | 0.9653 | 0.0757 |
| Semantic guard probe | 0.8900 | 0.9678 | 0.2150 |

Semantic guard probe 没有降低 ImageNet 子集 kNN Top-1，同时类中心间隔明显增加，说明该方向比 full ATAS 更好地保留了全局语义。

### VOC2012

| 设置 | 模型 | 前景 mIoU | 前景像素准确率 | 平均类别准确率 |
| --- | --- | ---: | ---: | ---: |
| Vanilla | OpenCLIP baseline | 0.4016 | 0.5565 | 0.5680 |
| Vanilla | Full ATAS epoch 6 | 0.3029 | 0.4672 | 0.5152 |
| Vanilla | Semantic guard probe | 0.3800 | 0.5329 | 0.5665 |
| SCLIP 风格 | OpenCLIP baseline | 0.6826 | 0.8034 | 0.8292 |
| SCLIP 风格 | Full ATAS epoch 6 | 0.4244 | 0.5618 | 0.6388 |
| SCLIP 风格 | Semantic guard probe | 0.6614 | 0.7772 | 0.8104 |

结论：

- Semantic guard probe 明显优于 full ATAS epoch 6，说明保守训练方向有效。
- 但它仍低于 OpenCLIP baseline，因此还不能认为已经复现出作者论文中的 ATAS 增益。
- 后续应继续围绕“减少 patch token 破坏，同时保留 ATAS 局部迁移能力”做消融，而不是直接重复完整 ImageNet 训练。

## 与作者论文结果的关系

作者论文报告的 VOC20 结果中：

- Vanilla CLIP：41.8 mIoU
- Vanilla ATAS：56.0 mIoU
- SCLIP CLIP：78.2 mIoU
- SCLIP ATAS：80.6 mIoU

我们当前结果显示：

- vanilla OpenCLIP baseline 为 40.16 mIoU，和作者 Vanilla CLIP baseline 接近。
- SCLIP 风格 OpenCLIP baseline 为 68.26 mIoU，低于作者 SCLIP baseline，但方向合理。
- ATAS checkpoint 没有带来论文中的 dense prediction 增益。

因此，当前阶段最稳妥的报告口径不是“完全复现了论文提升”，而是：

1. 已经完整搭建并跑通了作者设置的 ATAS 自蒸馏训练流程。
2. 完整 ImageNet 6 epoch checkpoint 已生成。
3. 在 VOC2012 下游评估中，我们的 ATAS checkpoint 没有复现出论文的 mIoU 提升。
4. 结果提示关键差异可能来自实现细节，包括 teacher/student token 对齐、mosaic 训练细节、dense evaluation 框架细节，或 OpenCLIP 与论文 CLIP 权重/实现差异。

## 后续优先排查方向

接下来如果继续提高复现程度，优先做：

1. 检查 ATAS 训练损失实现是否与论文完全一致，尤其是 GLD/LLD/GGD 的正负样本构造和归一化位置。
2. 对比 OpenCLIP 与论文使用的 CLIP ViT-B/16 权重和视觉 transformer forward 细节。
3. 接入官方或成熟的 MaskCLIP/SCLIP 代码，而不是当前轻量近似实现。
4. 额外做 ImageNet kNN 或 retrieval 评估，确认完整 ImageNet 训练是否保持了全局语义能力。
5. 沿 semantic guard 方向继续扫 `GLD/GGD/lr`，先在子集和 VOC 快速筛选，再决定是否启动完整 ImageNet 训练。
