# 作者参数 QuickGELU 复现实验审计

本文档记录截至 2026-06-01 的 ATAS 复现实验审计结论。审计对象是作者参数对齐的 QuickGELU 版本：

```text
configs/atas_vitb_imagenet_full_author_quickgelu_2gpu_accum2.yaml
outputs/atas_vitb_imagenet_full_author_quickgelu_2gpu_accum2/checkpoint_epoch_6.pt
```

## 1. 实验配置

本轮训练使用 OpenAI CLIP ViT-B/16 权重，并显式启用 `force_quick_gelu=True`。关键设置如下：

| 项目 | 设置 |
| --- | --- |
| 训练数据 | ImageNet-1K train 全量 |
| 模型 | OpenCLIP `ViT-B-16` |
| 激活函数 | QuickGELU |
| global 输入 | `224x224` |
| mosaic 输入 | `960x960` |
| mosaic grid | `6x6` |
| epoch | `6` |
| 优化器 | AdamW |
| 学习率 | `1e-5` |
| weight decay | `0.1` |
| 每卡 batch size | `36` |
| GPU | 2 卡 |
| 梯度累积 | `2` |
| 等效优化 batch | `36 x 2 GPU x 2 accum = 144` |
| 损失权重 | `GLD=1.0, LLD=0.01, GGD=1.0` |
| LLD patch 采样 | `max_lld_patches=1024` |

训练完成时间：

```text
checkpoint_epoch_6.pt: 2026-05-30 16:09
post evaluation finished: 2026-05-30 16:15
```

## 2. VOC2012 评估结果

本轮结果与 QuickGELU baseline 对比如下：

| 评估方式 | 模型 | Foreground mIoU | Pixel Acc | Mean Class Acc | 相对 baseline |
| --- | --- | ---: | ---: | ---: | ---: |
| Vanilla patch matching | QuickGELU CLIP baseline | 0.3604 | 0.5191 | 0.5111 | - |
| Vanilla patch matching | ATAS epoch 6 | 0.3111 | 0.4729 | 0.5240 | -0.0493 |
| SCLIP 风格 | QuickGELU CLIP baseline | 0.7650 | 0.8602 | 0.8790 | - |
| SCLIP 风格 | ATAS epoch 6 | 0.5703 | 0.7024 | 0.7750 | -0.1947 |

结论：

- QuickGELU 修正确实提升了 baseline，尤其 SCLIP baseline 从旧实验的 `0.6826` 提升到 `0.7650`。
- 但作者参数 ATAS epoch 6 仍低于 QuickGELU baseline。
- 这说明当前代码已经更接近作者使用的 CLIP 激活函数设置，但还没有复现论文宣称的 ATAS 增益。

## 3. 表征漂移诊断

诊断输出：

```text
outputs/checkpoint_drift_full_author_quickgelu_2gpu_accum2_epoch6/metrics.md
```

关键指标：

| 模型 | CLS cosine | Global patch cosine | Mosaic patch cosine | CLS pairwise MSE | Mosaic patch pairwise MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| QuickGELU baseline | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| ATAS epoch 6 | 0.6816 | 0.5106 | -0.0393 | 0.0510 | 0.5448 |

解释：

- `CLS cosine=0.6816` 说明 student 的全局语义已经明显偏离 teacher。
- `Global patch cosine=0.5106` 表明普通图像 patch token 也发生了较大变化。
- `Mosaic patch cosine=-0.0393` 是最关键的问题，说明 mosaic patch token 与 teacher 几乎失去正相关。
- `Mosaic patch pairwise MSE=0.5448` 表明局部 patch-patch 相似度结构没有被很好保留。

当前最合理的判断是：本实现中作者参数会造成过强的局部表征漂移，导致 downstream dense prediction 没有获得论文中的提升。

## 4. 三项蒸馏模块实现核对

### GLD：全局到局部蒸馏

代码位置：

```text
src/atas/losses.py
train_atas.py
```

实现路径：

```python
loss_gld = region_global_to_local_loss(
    student_patches=student_mosaic_patches,
    teacher_cls=teacher_region_cls,
    region_boxes=region_boxes,
    patch_grid=patch_grid,
    temperature=config.temperature,
)
```

实现逻辑：

1. `MosaicBatchCollator` 记录每个 mosaic cell 对应的源图像位置。
2. teacher 对原始 `224x224` 图像提取 global CLS。
3. student 对 `960x960` mosaic 图像提取 patch tokens。
4. 对每个 cell 内的 patch tokens 与对应 teacher CLS 计算相似度。
5. 使用 softmax 权重做 region pooling。
6. 用 InfoNCE 将 pooled local region 对齐到 teacher CLS。

审计结论：

- GLD 的主结构与论文描述一致。
- 当前实现是 cell-level region pooling，不是像素级或任意 mask-level 区域。
- 当前 InfoNCE 只使用当前进程本地 batch 的负样本，没有跨 GPU 收集负样本。

### LLD：局部到局部蒸馏

代码位置：

```text
src/atas/losses.py::local_to_local_loss()
```

实现逻辑：

1. teacher 和 student 都对同一张 mosaic 图像提取 patch tokens。
2. 分别对 patch tokens 做 L2 normalize。
3. 计算 patch-patch cosine similarity matrix。
4. 用 MSE 约束 student 的局部相似度结构接近 teacher。

审计结论：

- LLD 的 pairwise structure preservation 逻辑与论文目标一致。
- 为控制 `960x960` 输入下的显存，当前只随机采样 `1024` 个 patch 计算 LLD。
- 由于 `lambda_lld=0.01`，实际加权后的 LLD 对总损失影响很小，可能不足以约束 patch 漂移。

### GGD：全局到全局蒸馏

代码位置：

```text
src/atas/losses.py::global_to_global_loss()
```

实现逻辑：

```python
loss_ggd = global_to_global_loss(student_global_cls, teacher_global_cls, config.temperature)
```

GGD 直接用 student global CLS 和 teacher global CLS 做 batch 内 InfoNCE。正样本是同一图像的 teacher CLS，负样本是 batch 内其他图像的 teacher CLS。

审计结论：

- GGD 的主目标与论文一致，用于保留全局 CLIP 语义。
- 当前实现没有跨 GPU `all_gather`，因此多卡训练时每张卡只看到本地负样本。
- 如果作者实现使用全局 batch 负样本，本项目当前 GGD/GLD 的对比学习强度会低于作者实现。

## 5. Teacher/Student 与 QuickGELU 核对

当前训练和评估均从配置读取：

```yaml
model:
  name: ViT-B-16
  pretrained: /mnt/t1b6/xuzhejia/checkpoints/open_clip/open_clip_pytorch_model.bin
  quick_gelu: true
```

训练代码中：

```python
student, _, _ = open_clip.create_model_and_transforms(
    model_name,
    pretrained=pretrained,
    force_quick_gelu=force_quick_gelu,
)
teacher = copy.deepcopy(student)
teacher.eval()
for parameter in teacher.parameters():
    parameter.requires_grad_(False)
```

审计结论：

- QuickGELU 已经正确启用。
- teacher 由初始 student 深拷贝得到，并被冻结。
- 训练只更新 `student.visual`。
- checkpoint 只保存 `student.visual`，评估时再加载到同构 CLIP visual encoder。

## 6. 评估实现核对

VOC2012 评估代码：

```text
scripts/evaluate_voc_zero_shot_seg.py
```

当前实现包括：

- `vanilla`：直接使用 ViT final patch tokens 与 VOC 文本特征匹配。
- `sclip`：在最后一层 ViT block 使用 self-correlation attention 的轻量 SCLIP 风格实现。
- 评估只统计 VOC 前景 20 类，不显式建模 background。

审计结论：

- 当前评估足以作为轻量 zero-shot dense prediction probe。
- 但它不是作者完整 dense prediction pipeline。
- SCLIP 部分是复现式近似实现，不等价于官方 SCLIP/MaskCLIP 代码。

## 7. 当前与作者结果差距

论文报告的 VOC20 结果：

| 方法 | 论文 mIoU |
| --- | ---: |
| Vanilla CLIP | 41.8 |
| Vanilla ATAS | 56.0 |
| SCLIP CLIP | 78.2 |
| SCLIP ATAS | 80.6 |

当前 QuickGELU 实验：

| 方法 | 当前 mIoU |
| --- | ---: |
| Vanilla QuickGELU CLIP | 36.0 |
| Vanilla QuickGELU ATAS | 31.1 |
| SCLIP QuickGELU CLIP | 76.5 |
| SCLIP QuickGELU ATAS | 57.0 |

口径建议：

> 本项目已实现 ATAS 的 GLD、LLD、GGD 三项核心蒸馏，并完成 ImageNet 全量作者参数训练；但在当前 VOC2012 zero-shot dense evaluation 下，ATAS checkpoint 未复现论文中报告的 mIoU 提升。表征诊断显示 student patch token 漂移过大，是当前失败的主要技术证据。

## 8. 主要风险点与后续优先级

审计后已完成第一项实现修正：GLD/GGD 已支持 DDP `all_gather` teacher features。启用方式：

```yaml
training:
  gather_distributed_negatives: true
```

### 修正后的 probe 结果

| 设置 | Vanilla mIoU | SCLIP mIoU | kNN Top-1 | Mosaic patch cosine | Patch pairwise MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| QuickGELU all-gather probe | 0.3615 | 0.6901 | 0.9083 | 0.0595 | 0.3762 |
| QuickGELU semantic guard + all-gather probe | 0.3522 | 0.7410 | 0.9113 | 0.4667 | 0.0257 |

结论：

- all-gather 在子集 probe 中明显改善结果，说明“本地负样本不足”是一个重要实现差异。
- semantic guard 与 all-gather 组合最稳定，SCLIP mIoU 达到 `0.7410`，距离 QuickGELU SCLIP baseline `0.7650` 已经较近。
- 但 semantic guard 修改了作者给出的损失权重，因此它更适合作为诊断和改进方向，而不是作者参数完全复现结果。

### 完整 ImageNet b72 all-gather 结果

`configs/atas_vitb_imagenet_full_author_quickgelu_2gpu_b72_allgather.yaml` 已完成 6 epoch 训练。该配置使用 2GPU、每卡 batch72，使全局负样本数和等效优化 batch 都为 144。

| Dense mode | Foreground mIoU | Pixel Acc | Mean Class Acc |
| --- | ---: | ---: | ---: |
| Vanilla | 0.3090 | 0.4699 | 0.5198 |
| SCLIP 风格 | 0.5817 | 0.7011 | 0.7803 |

| 模型 | CLS cosine | Global patch cosine | Mosaic patch cosine | Mosaic patch pairwise MSE |
| --- | ---: | ---: | ---: | ---: |
| QuickGELU baseline | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| QuickGELU b72 all-gather epoch 6 | 0.6876 | 0.5388 | -0.0513 | 0.4838 |

该结果说明：补齐全局负样本后，完整作者参数训练仍没有接近作者宣称效果。SCLIP mIoU 只从 `0.5703` 小幅提升到 `0.5817`，mosaic patch cosine 仍为负值，核心问题仍是 patch token 漂移。

### 后续优先级

1. **完整 ImageNet semantic guard + all-gather**：把子集上有效的 semantic guard 组合扩展到完整 ImageNet，但报告中必须标注它已经不是作者原始参数。
2. **LLD 约束强度**：验证完整 patch-patch 或更稳定采样策略，确认 `lambda_lld=0.01` 下是否足以保留 patch 结构。
3. **学习率调度**：当前训练没有显式 warmup/cosine scheduler；如果作者实现使用 scheduler，需要补齐。
4. **官方 dense evaluation**：接入更接近作者的 MaskCLIP/SCLIP 评估实现，避免评估管线差异掩盖训练效果。
5. **复现报告表述**：严格作者参数复现目前未达标，应把 all-gather、semantic guard 等结果作为失败诊断和改进路线，而不是声称已经复现论文指标。
