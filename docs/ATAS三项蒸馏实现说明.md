# ATAS 三项蒸馏模块实现说明

本文档说明本复现代码中如何实现论文 **ATAS: Any-to-Any Self-Distillation for Enhanced Open-Vocabulary Dense Prediction** 的三个核心模块：

1. 全局到局部蒸馏，Global-to-Local Distillation，简称 `GLD`。
2. 局部到局部蒸馏，Local-to-Local Distillation，简称 `LLD`。
3. 全局到全局蒸馏，Global-to-Global Distillation，简称 `GGD`。

对应代码主要位于：

```text
src/atas/losses.py
train_atas.py
```

当前作者参数对齐配置为：

```text
configs/atas_vitb_imagenet_full_author_quickgelu_2gpu_accum2.yaml
```

其关键参数保持为：

```text
lambda_gld = 1.0
lambda_lld = 0.01
lambda_ggd = 1.0
learning_rate = 1e-5
weight_decay = 0.1
temperature = 1.0
mosaic_choices = [6]
epochs = 6
batch_size = 36 per GPU
gradient_accumulation_steps = 2
quick_gelu = true
```

该配置使用 2 卡 DDP 加梯度累积，等效优化 batch 为 `36 x 2 x 2 = 144`。需要注意：这个历史实验的 InfoNCE 负样本仍来自每个 DDP 进程的本地 batch，梯度累积不会扩大单次对比学习的负样本集合。

后续代码已新增跨 GPU 负样本支持：

```yaml
training:
  gather_distributed_negatives: true
```

启用后，GLD/GGD 会在 DDP 进程间 `all_gather` teacher features，使 student query 与全局 batch 的 teacher keys 对比。

## 1. 总体训练流程

训练入口是 `train_atas.py`。

### Teacher 与 Student

代码中先加载一份 OpenCLIP ViT-B/16：

```python
student, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
teacher = copy.deepcopy(student)
```

其中：

- `teacher` 是冻结的原始 CLIP 图像编码器。
- `student` 是被 ATAS 训练更新的图像编码器。
- 文本编码器不参与训练。
- 训练时只保存 `student.visual` 的参数。

冻结 teacher 的代码：

```python
teacher.eval()
for parameter in teacher.parameters():
    parameter.requires_grad_(False)
```

### Global 图像与 Mosaic 图像

`MosaicBatchCollator` 同时构造两类输入：

```python
global_images
mosaic_images
region_boxes
```

含义如下：

- `global_images`：每张原始 ImageNet 图像 resize 到 `224x224`，用于抽取全局 CLS token。
- `mosaic_images`：把一个 batch 内的图像拼成 `6x6` mosaic，再 resize 到 `960x960`，用于抽取局部 patch token。
- `region_boxes`：记录 mosaic 中每个 cell 对应的源图像位置，用于把局部 patch 区域和对应的全局 teacher CLS token 对齐。

在作者参数配置下：

```text
image_size = 960
mosaic grid = 6x6
patch size = 16
mosaic patch grid = 60x60
每个 cell = 10x10 patch
```

### Token 抽取

`encode_visual_tokens()` 显式执行 OpenCLIP ViT 的视觉 forward，并返回：

```python
cls_token, patch_tokens
```

训练时抽取四组特征：

```python
teacher_global_cls, _ = teacher_encoder(global_images)
_, teacher_mosaic_patches = teacher_encoder(mosaic_images)
student_global_cls, _ = student_encoder(global_images)
_, student_mosaic_patches = student_encoder(mosaic_images)
```

然后送入：

```python
atas_region_loss(...)
```

## 2. GLD：全局到局部蒸馏

### 论文目标

GLD 的目标是把 teacher 的全局 CLS token 中已经对齐到文本空间的语义，迁移到 student 的局部 patch token，使局部区域也获得更强的语义 grounding。

论文中的核心形式可以概括为：

```text
用 teacher CLS token 和 student patch token 的相似度作为权重，
对 student patch token 做加权池化，
再用对比学习把这个局部聚合表示对齐到 teacher CLS token。
```

### 本代码实现位置

对应函数：

```text
src/atas/losses.py
weighted_region_pool()
region_global_to_local_loss()
global_to_local_loss()
```

完整训练使用的是 mosaic 版本：

```python
loss_gld = region_global_to_local_loss(
    student_patches=student_mosaic_patches,
    teacher_cls=teacher_region_cls,
    region_boxes=region_boxes,
    patch_grid=patch_grid,
    temperature=config.temperature,
)
```

### 具体实现

对于每个 mosaic cell：

1. 根据 `region_boxes` 取出该 cell 内的 student patch token。
2. 对 patch token 和对应源图像的 teacher CLS token 做 L2 normalize。
3. 计算每个 patch 与 teacher CLS 的 cosine similarity。
4. 对 similarity 除以 temperature 后做 softmax，得到 patch 权重。
5. 用权重对 patch token 做加权池化，得到局部聚合表示。
6. 用 batch 内对比学习，把该局部聚合表示对齐到对应的 teacher CLS token。

代码核心：

```python
region = normalize_features(region)
weights = region @ teacher_cls[index]
weights = F.softmax(weights / temperature, dim=0)
pooled_regions.append(torch.sum(weights[:, None] * region, dim=0))
```

随后：

```python
return contrastive_self_distill(pooled_regions, teacher_cls, temperature)
```

其中 `contrastive_self_distill()` 的实现是标准 batch 内 InfoNCE：

```python
student = normalize_features(student)
teacher = normalize_features(teacher)
logits = student @ teacher.t()
logits = logits / temperature
targets = torch.arange(logits.shape[0], device=logits.device)
return F.cross_entropy(logits, targets)
```

因此正样本是同一图像或同一 mosaic cell 对应的 teacher CLS，负样本是 batch 内其他 teacher CLS。

在 `gather_distributed_negatives=true` 时，负样本来自所有 DDP rank 的 teacher CLS。当前实现仍只对本 rank 的 student query 计算 loss，但 logits 的列扩展为全局 teacher batch，target offset 根据 `rank * local_batch` 修正。

## 3. LLD：局部到局部蒸馏

### 论文目标

LLD 的目标是保持 CLIP 原始 patch token 之间的局部语义结构。换句话说，student 可以调整 patch token 以获得更好的局部语义对齐，但不应破坏 teacher 中 patch-patch 的相似度关系。

### 本代码实现位置

对应函数：

```text
src/atas/losses.py
local_to_local_loss()
```

训练中调用位置：

```python
loss_lld = local_to_local_loss(student_lld_patches, teacher_lld_patches)
```

### 具体实现

对同一张 mosaic 图像：

1. 取 student 的 mosaic patch token。
2. 取 teacher 的 mosaic patch token。
3. 对两者分别做 L2 normalize。
4. 分别计算 patch-patch cosine similarity matrix。
5. 用 MSE 约束 student 的 patch-patch 关系接近 teacher 的 patch-patch 关系。

代码核心：

```python
student_patches = normalize_features(student_patches)
teacher_patches = normalize_features(teacher_patches)

student_rel = torch.bmm(student_patches, student_patches.transpose(1, 2))
teacher_rel = torch.bmm(teacher_patches, teacher_patches.transpose(1, 2))
return F.mse_loss(student_rel, teacher_rel)
```

为了控制 `960x960` mosaic 下的显存和计算量，配置中使用：

```text
max_lld_patches = 1024
```

当 patch 数超过该值时，代码会随机采样一部分 patch 计算 LLD：

```python
indices = torch.randperm(student_mosaic_patches.shape[1], device=student_mosaic_patches.device)
indices = indices[:max_lld_patches]
student_lld_patches = student_mosaic_patches[:, indices]
teacher_lld_patches = teacher_mosaic_patches[:, indices]
```

这与完整 patch-patch 矩阵相比是一个工程近似，目的是让完整 ImageNet 训练可以在 A6000 上稳定运行。

## 4. GGD：全局到全局蒸馏

### 论文目标

GGD 的目标是防止训练过程中 student 的全局语义能力退化。GLD 会推动局部 token 适应 dense prediction，但如果没有全局约束，student 的 CLS token 可能偏离原始 CLIP 的 image-level 语义。

### 本代码实现位置

对应函数：

```text
src/atas/losses.py
global_to_global_loss()
contrastive_self_distill()
```

训练中调用位置：

```python
loss_ggd = global_to_global_loss(student_global_cls, teacher_global_cls, config.temperature)
```

### 具体实现

GGD 直接使用 batch 内对比学习：

```python
return contrastive_self_distill(student_cls, teacher_cls, temperature)
```

其逻辑是：

- query：student 的 global CLS token。
- positive key：同一图像的 teacher global CLS token。
- negative keys：batch 内其他图像的 teacher global CLS token。

这会让 student 的全局表示继续靠近原始 CLIP teacher，从而保留 image-level 分类和检索语义。

## 5. 总损失

三项损失在 `atas_region_loss()` 中汇总：

```python
total = (
    config.lambda_gld * loss_gld
    + config.lambda_lld * loss_lld
    + config.lambda_ggd * loss_ggd
)
```

作者参数对齐配置为：

```text
lambda_gld = 1.0
lambda_lld = 0.01
lambda_ggd = 1.0
```

训练日志中会输出：

```text
loss
loss_gld
loss_lld
loss_ggd
```

用于观察三项损失是否稳定。

## 6. 当前复现结果与问题定位

在作者参数一致的完整 ImageNet 训练中，我们已经完成：

```text
outputs/atas_vitb_imagenet_full_author_quickgelu_2gpu_accum2/checkpoint_epoch_6.pt
```

但 VOC2012 下游结果没有达到作者论文宣称的提升：

| 设置 | OpenCLIP baseline | ATAS epoch 6 |
| --- | ---: | ---: |
| Vanilla VOC foreground mIoU | 0.3604 | 0.3111 |
| SCLIP 风格 VOC foreground mIoU | 0.7650 | 0.5703 |

表征漂移诊断显示：

| 模型 | CLS cosine | Global patch cosine | Mosaic patch cosine | Patch pairwise MSE |
| --- | ---: | ---: | ---: | ---: |
| OpenCLIP baseline | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| QuickGELU ATAS epoch 6 | 0.6816 | 0.5106 | -0.0393 | 0.5448 |

这说明在当前实现中，作者参数下训练得到的 student patch token 与 teacher patch token 发生了明显偏移。该偏移可能是下游 VOC 指标下降的重要原因。

## 7. 仍可能存在的实现差异

当前代码已经按论文公式实现三项核心蒸馏，但仍可能与作者实现存在以下差异：

1. OpenCLIP 与作者使用的 CLIP 权重或 forward 细节不完全一致。
2. `960x960` mosaic 下的 positional embedding resize 方式可能与作者不同。
3. LLD 使用 `max_lld_patches=1024` 采样近似，而非完整 patch-patch 矩阵。
4. 历史完整实验没有跨 GPU `all_gather` 负样本；当前代码已补充该功能，正在运行 batch72 all-gather 的完整 ImageNet 实验。
5. 当前训练没有显式 warmup/cosine 学习率调度；如果作者实现包含 scheduler，这会造成训练动力学差异。
6. 本项目的 VOC vanilla/SCLIP 评估是轻量复现，不等同于作者完整 dense prediction pipeline。
7. 论文未公开代码时，teacher/student token 具体取层、归一化位置、mosaic 细节仍可能存在隐含实现差异。

因此，当前结论应表述为：

> 本项目实现并跑通了 ATAS 的三项蒸馏模块，并保持作者论文给出的主要训练参数；但当前代码在 VOC2012 下游评估中未复现论文宣称的性能提升。实验诊断显示，作者参数下 student patch token 漂移过大，是后续排查的核心问题。
