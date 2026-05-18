# ATAS 论文复现项目

本项目用于复现 ICCV 2025 论文 **ATAS: Any-to-Any Self-Distillation for Enhanced Open-Vocabulary Dense Prediction**。

课程复现的重点不是完整复刻论文所有大规模实验，而是尽量完整地复现核心方法，并给出可解释、可汇报的实验链条：训练实现、定量评估、消融实验、patch 级可视化，以及轻量级密集预测代理评估。

## 论文核心思想

ATAS 的目标是增强 CLIP ViT 图像编码器的局部 patch 表征，使它更适合开放词表密集预测任务，例如语义分割和目标检测。

方法上，ATAS 使用冻结的原始 CLIP 图像编码器作为教师模型，通过自蒸馏训练学生图像编码器。训练中主要包含三个损失函数：

- `GLD`：Global-to-Local Distillation，全局到局部蒸馏，让 patch token 对齐全局语义。
- `LLD`：Local-to-Local Distillation，局部到局部蒸馏，让不同视角或 mosaic 中的局部区域保持一致。
- `GGD`：Global-to-Global Distillation，全局到全局蒸馏，用来稳定 CLIP 原有的图像级语义能力。

我们目前的复现路线是：先训练 ATAS 风格的 CLIP ViT-B/16，再用 kNN、消融、patch 可视化和零样本分割代理任务来验证方法是否有效。

## 实验环境

- 服务器：实验室 A6000 服务器。
- GPU：4 x NVIDIA RTX A6000，每张 48GB 显存。
- 系统：Ubuntu 20.04。
- Conda 环境：`atas`。
- 服务器项目目录：`/mnt/t1b6/xuzhejia/atas-repro`。
- 数据集目录：`/mnt/t1b6/xuzhejia/datasets`。

## 数据准备情况

已完成：

- 完整 ImageNet train 已经在服务器上解压。
- 已构建课程规模子集：`ImageNet-100x200`，即 100 个类别，每类 200 张图，共 20,000 张图。
- 子集路径：`/mnt/t1b6/xuzhejia/datasets/imagenet_subset_100x200/train`。

VOC2012：

- VOC2012 已由本地文件 `F:\dataset\VOCtrainval_11-May-2012.tar` 上传到服务器并解压。
- VOC2012 解压目录：`/mnt/t1b6/xuzhejia/datasets/VOCdevkit/VOC2012`。
- VOC2012 val split 共 1449 张图，已完成零样本分割评估。

## 已实现内容

训练代码已经完成以下功能：

- OpenCLIP ViT-B/16 骨干网络加载。
- Mosaic 图像构造。
- `GLD`、`LLD`、`GGD` 三个核心损失函数。
- AMP 混合精度训练。
- 检查点保存与恢复。
- 训练配置文件管理。
- kNN 评估脚本。
- 消融实验配置。
- patch alignment 可视化脚本。
- VOC-like 零样本分割评估脚本。
- VOC2012 零样本分割评估脚本。

重要修复：

- 修复了 OpenCLIP ViT token 提取路径中的维度顺序问题。
- 修复后，自定义 `encode_visual_tokens()` 与 OpenCLIP 原生 `encode_image()` 的行为对齐，避免 CLS token 表征异常塌缩。

## 主实验进展

当前稳定训练配置：

```text
configs/atas_vitb_subset_100x200_stable.yaml
```

训练设置：

- Backbone：OpenCLIP ViT-B/16。
- 训练数据：ImageNet-100x200 子集。
- Mosaic：`6x6`。
- batch size：36。
- 训练轮数：6 epochs。

当前稳定检查点：

```text
outputs/atas_vitb_subset_100x200_stable/checkpoint_epoch_6.pt
```

`checkpoint_epoch_3.pt` 保留为中间阶段结果，主要用于消融对比和部分 dense proxy 分析。

## kNN 图像级评估结果

评估方式：

- 使用 100 类 ImageNet 子集。
- 每类 160 张作为检索库。
- 每类 40 张作为查询集。
- 指标包括 Top-1 kNN、Top-5 命中率和类中心间隔。

结果如下：

| 模型 | Top-1 kNN | Top-5 命中率 | 类中心间隔 |
| --- | ---: | ---: | ---: |
| 原始 OpenCLIP | 0.8900 | 0.9653 | 0.0757 |
| ATAS 第 3 轮 | 0.8913 | 0.9618 | 0.2239 |
| ATAS 第 6 轮 | 0.8965 | 0.9650 | 0.2252 |

结论：

- ATAS 训练到第 6 轮后 Top-1 kNN 提升到 0.8965，说明图像级语义能力没有被破坏，并在当前子集上有所增强。
- 类中心间隔从 0.0757 提升到 0.2252，说明类别中心分离更明显。
- 这说明当前检查点不是简单复制原始 CLIP，而是在保持全局语义的同时改变了特征空间结构。

结果目录：

```text
outputs/eval_subset_100x200_knn/
outputs/eval_subset_100x200_epoch6_knn/
```

## 消融实验结果

为了验证三个损失函数的作用，我们训练了 `GLD only`、`GLD + LLD` 和 `Full ATAS` 三组模型。

| 实验 | GLD | LLD | GGD | Top-1 kNN | Top-5 命中率 | 类中心间隔 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 原始 OpenCLIP | 否 | 否 | 否 | 0.8900 | 0.9653 | 0.0757 |
| GLD only | 是 | 否 | 否 | 0.2412 | 0.5070 | -0.0069 |
| GLD + LLD | 是 | 是 | 否 | 0.2117 | 0.4542 | -0.0152 |
| Full ATAS | 是 | 是 | 是 | 0.8913 | 0.9618 | 0.2239 |

结论：

- 只使用 `GLD` 或 `GLD + LLD` 会明显破坏原始 CLIP 的图像级语义能力。
- 加入 `GGD` 后，Full ATAS 能基本保持原始 CLIP 的 Top-1 kNN，同时显著提升 centroid margin。
- 这个现象支持论文中的设计动机：`GLD/LLD` 强化局部表征，`GGD` 负责稳定全局语义。

结果目录：

```text
outputs/ablation_summary/
```

## Patch 级可视化结果

已完成 patch 到 CLS 的对齐可视化，用来比较原始 OpenCLIP 和 ATAS 检查点的局部 patch 响应差异。

可视化脚本：

```text
scripts/visualize_patch_alignment.py
```

结果目录：

```text
outputs/patch_alignment_vis_full_448/
outputs/patch_alignment_vis_epoch6_448/
```

当前已生成 epoch 3 和 epoch 6 两组 448 分辨率可视化图。它们可以作为 PPT 中的定性结果，用于说明 ATAS 训练后局部 patch 响应发生了变化。

## MSD 零样本分割代理评估

为了补充密集预测风格的定量结果，我们使用服务器上已有的 MSD 手机屏幕缺陷分割数据集做了零样本分割代理实验。

注意：MSD 不是论文标准评测集，只是当前阶段的密集预测代理实验，用来验证 patch token 与文本类别的对齐趋势。

数据设置：

- 数据集：`/data/haier/wangcairui/dataset/MSD`
- 数据划分：`val`
- 图像数：240
- 类别：`oil stain`、`stain`、`scratch`
- 输入分辨率：448
- 预测方式：ViT patch token 与 CLIP text embedding 做相似度匹配，再上采样到像素级。

前景三分类结果如下：

| 模型 | 前景 mIoU | 前景像素准确率 | 平均类别准确率 | Oil IoU | Stain IoU | Scratch IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始 OpenCLIP | 0.0940 | 0.2059 | 0.3143 | 0.1809 | 0.0338 | 0.0674 |
| ATAS 第 3 轮 | 0.2575 | 0.7428 | 0.3124 | 0.7454 | 0.0018 | 0.0252 |

加入背景类后的严格结果如下：

| 模型 | mIoU | 像素准确率 | 前景 mIoU | 前景像素准确率 | 背景 IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始 OpenCLIP | 0.1623 | 0.6470 | 0.0006 | 0.0524 | 0.6474 |
| ATAS 第 3 轮 | 0.1255 | 0.4989 | 0.0014 | 0.2598 | 0.4980 |

结论：

- 在前景三分类代理任务中，ATAS 的前景 mIoU 和前景像素准确率明显高于原始 OpenCLIP。
- 提升主要来自 Oil 类，Stain 和 Scratch 仍然不稳定。
- 加入背景类后，结果对背景提示词非常敏感，不能作为主要正向证据。
- 因此，MSD 实验更适合在报告中作为“密集预测代理验证 + 局限性分析”。

结果目录：

```text
outputs/msd_zero_shot_seg_baseline_balanced/
outputs/msd_zero_shot_seg_full_atas_balanced/
outputs/msd_zero_shot_seg_baseline_with_bg/
outputs/msd_zero_shot_seg_full_atas_with_bg/
```

详细记录：

```text
docs/MSD_zero_shot_segmentation_proxy.md
```

## VOC2012 零样本分割评估

在本地下载 VOC2012 后，我们已经把数据上传到服务器，并完成了 VOC2012 val split 的零样本分割评估。

数据设置：

- 数据集：PASCAL VOC2012 trainval。
- 服务器路径：`/mnt/t1b6/xuzhejia/datasets/VOCdevkit/VOC2012`。
- 验证图像数：1449。
- 类别数：20 个前景类别。
- 评估方式：ViT patch token 与 VOC 类别文本特征直接匹配，再上采样到像素级。

结果如下：

| 模型 | 前景 mIoU | 前景像素准确率 | 平均类别准确率 |
| --- | ---: | ---: | ---: |
| 原始 OpenCLIP | 0.4016 | 0.5565 | 0.5680 |
| ATAS 第 3 轮 | 0.3604 | 0.5174 | 0.5789 |
| ATAS 第 6 轮 | 0.3287 | 0.4854 | 0.5462 |

结论：

- 在这个更标准的 VOC2012 zero-shot patch matching 评估上，ATAS 第 3 轮和第 6 轮的前景 mIoU 都低于原始 OpenCLIP。
- ATAS 第 3 轮的平均类别准确率略高，且在 `car`、`horse`、`bus`、`dining table`、`boat` 等类别上有提升；第 6 轮虽然 ImageNet kNN 更好，但 VOC 指标进一步下降。
- 这说明当前课程规模训练还不足以稳定超过原始 CLIP，完整 dense prediction 复现仍需要更大训练规模和更完整的下游分割框架。
- 这组结果适合作为“标准数据集补充实验与局限性分析”，不要作为主要正向结果。

结果目录：

```text
outputs/voc_zero_shot_seg_baseline/
outputs/voc_zero_shot_seg_full_atas/
outputs/voc_zero_shot_seg_full_atas_epoch6/
```

详细记录：

```text
docs/VOC2012零样本分割评估结果.md
```

## 作者设置对齐训练

为了向论文主结果靠近，当前已经启动完整 ImageNet 训练，而不是继续延长 2 万张子集训练。

当前正式配置：

```text
configs/atas_vitb_imagenet_full_author.yaml
```

该配置对齐作者论文中的主要训练设置：完整 ImageNet、6 epochs、4 卡训练、batch size 36 per GPU、AdamW、学习率 1e-5、weight decay 0.1、`GLD=1, LLD=0.01, GGD=1`。

当前状态：

- DDP smoke test 已通过。
- 完整 ImageNet 4 卡训练已启动。
- 训练日志：`/mnt/t1b6/xuzhejia/logs/atas_full_author_wait.log`。
- 输出目录：`outputs/atas_vitb_imagenet_full_author/`。
- checkpoint 产生后，将优先运行 `scripts/run_voc_full_author_sweep.sh` 做 VOC2012 vanilla 评估。
- 同时已预留 `scripts/run_voc_sclip_eval.sh`，用于后续做更接近论文下游框架的 SCLIP 风格评估。

同时，已经完成子集 checkpoint 的 VOC sweep。结果显示 epoch 1 到 epoch 6 均未超过 OpenCLIP baseline，且后续 epoch 在 VOC dense proxy 上逐步下降。这说明问题更可能来自训练规模和下游评估框架，而不是单纯训练轮数不足。

近似 MaskCLIP 尝试也已记录：当前简化实现结果异常，不能作为正式 MaskCLIP 复现结论；后续应接入更可靠的 MaskCLIP/SCLIP 实现。

详细记录：

```text
docs/完整ImageNet作者设置训练进展.md
docs/VOC_checkpoint_sweep与MaskCLIP尝试.md
```

## 当前复现结论

目前已经完成了课程大作业中比较完整的一条证据链：

1. 成功搭建 ATAS 训练流程。
2. 在 ImageNet 子集上训练得到稳定检查点。
3. kNN 结果表明模型保持了图像级语义能力。
4. 消融实验表明 `GGD` 是维持全局语义稳定性的关键。
5. patch 可视化展示了局部响应变化。
6. MSD 零样本分割代理实验补充了密集预测风格的定量与可视化结果。
7. VOC2012 零样本分割评估补充了更标准数据集上的结果和局限性分析。
8. 完整 ImageNet 作者设置训练已经启动，这是当前最接近论文主实验的路线。

目前最适合汇报的主结论：

- 我们复现了 ATAS 的核心自蒸馏训练机制。
- Full ATAS 在保持原始 CLIP 图像级语义能力的同时，提高了类别特征分离度；继续训练到 epoch 6 后，ImageNet 子集 Top-1 kNN 提升到 0.8965。
- 消融实验支持论文中 `GLD + LLD + GGD` 的组合设计。
- 当前密集预测代理结果显示 ATAS 对部分前景 patch-文本对齐有增强，但完整语义分割仍需要更标准的数据集和更稳的背景建模。

## 后续计划

短期继续做：

- 从 patch alignment、MSD 和 VOC2012 可视化中筛选最适合放进 PPT 的图。
- 整理最终实验报告中的方法、实验、消融、epoch 6 继续训练和局限性部分。

可选扩展：

- 接入 MaskCLIP 或 SCLIP 风格评估。
- 尝试更多 ImageNet 子集类别，验证 epoch 6 的 kNN 提升是否能在更大子集上保持。
- 针对背景提示词做更系统的提示词集成。

## 常用命令

环境检查：

```bash
python scripts/check_env.py
```

ATAS 训练：

```bash
python train_atas.py --config configs/atas_vitb_subset_100x200_stable.yaml
```

kNN 评估：

```bash
python scripts/evaluate_imagenet_subset_knn.py --config configs/atas_vitb_subset_100x200_stable.yaml
```

MSD 零样本分割代理评估：

```bash
GPU=3 OUTPUT_SUFFIX=_balanced SAVE_VIS=0 SAVE_VIS_PER_CLASS=4 bash scripts/run_msd_zero_shot_seg.sh
```

VOC2012 下载：

```bash
bash scripts/download_voc2012.sh
```

VOC2012 零样本分割评估：

```bash
GPU=2 bash scripts/run_voc_zero_shot_seg.sh
```

## 重要文档

- `docs/完整复现实验方案与进展.md`
- `docs/稳定子集训练实验记录.md`
- `docs/子集KNN评估结果.md`
- `docs/ATAS_epoch6继续训练与评估结果.md`
- `docs/后续实验计划.md`
- `docs/作者设置对齐实验计划.md`
- `docs/完整ImageNet作者设置训练进展.md`
- `docs/VOC_checkpoint_sweep与MaskCLIP尝试.md`
- `docs/Patch级可视化说明.md`
- `docs/MSD_zero_shot_segmentation_proxy.md`
- `docs/VOC2012零样本分割评估结果.md`
- `docs/当前复现进展汇报补充.md`
