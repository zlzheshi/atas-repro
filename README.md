# ATAS 复现项目

本仓库是论文 **ATAS: Any-to-Any Self-Distillation for Enhanced Open-Vocabulary Dense Prediction** 的 PyTorch 复现实现，重点覆盖 CLIP ViT 图像编码器自蒸馏训练，以及基于 PASCAL VOC2012 的开放词汇语义分割评估。

> 当前状态：已完成 ATAS 训练主流程、ImageNet 多卡训练、断点续训、VOC2012 vanilla patch matching 评估和 SCLIP 风格 dense inference 评估。最新一轮已修正 OpenAI CLIP ViT-B/16 的 QuickGELU 设置，但当前复现实验仍未达到论文中下游指标提升的效果。

## 特性

- 基于 OpenCLIP ViT-B/16 的 teacher-student 自蒸馏训练。
- 支持 6x6 mosaic 输入构造。
- 实现 ATAS 核心损失：`GLD`、`LLD`、`GGD`。
- 支持 AMP、checkpoint 保存/恢复、DDP 多卡训练。
- 支持 VOC2012 零样本语义分割评估。
- 支持 SCLIP 风格 self-correlation dense inference。
- 支持 DDP 下 GLD/GGD 跨 GPU all-gather 负样本。
- 保留最终实验结论文档和可复现实验配置。

## 项目结构

```text
configs/                 训练配置文件
scripts/                 数据准备、训练、评估和可视化脚本
src/                     项目公共模块
docs/                    课程汇报、服务器运行指南和实验结果记录
train_atas.py            ATAS 训练入口
requirements.txt         Python 依赖
```

大型数据集、模型权重、训练 checkpoint 和评估输出不提交到 git。

## 环境安装

建议使用 Python 3.10 或更高版本：

```bash
conda create -n atas python=3.10 -y
conda activate atas
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

检查环境：

```bash
python scripts/check_env.py
python scripts/test_openclip_load.py
```

## 数据准备

### ImageNet

训练脚本使用 ImageFolder 格式的 ImageNet 训练集：

```text
/path/to/imagenet/train/
  n01440764/
    image_1.JPEG
  n01443537/
    image_2.JPEG
```

ATAS 损失不使用监督标签，类别目录主要用于 PyTorch dataloader 读取数据。

### PASCAL VOC2012

VOC2012 用于零样本语义分割评估：

```text
/path/to/VOCdevkit/VOC2012/
  JPEGImages/
  SegmentationClass/
  ImageSets/Segmentation/val.txt
```

## 训练

### 调试训练

```bash
python train_atas.py \
  --config configs/atas_vitb_debug.yaml \
  --data-root /path/to/imagenet/train
```

### 完整 ImageNet 训练

本仓库最终保留两个完整训练配置：

```text
configs/atas_vitb_imagenet_full_author_quickgelu_2gpu_accum2.yaml
configs/atas_vitb_imagenet_full_author_quickgelu_2gpu_b72_allgather.yaml
```

二者都使用 OpenAI CLIP ViT-B/16、QuickGELU、完整 ImageNet、6 epochs、AdamW、学习率 `1e-5`、权重衰减 `0.1`，损失权重为 `GLD=1`、`LLD=0.01`、`GGD=1`。`b72_allgather` 是最终对齐全局负样本的版本。

实验室服务器上可使用等待空闲 GPU 的最终启动脚本：

```bash
bash scripts/run_full_imagenet_author_quickgelu_2gpu_b72_allgather_wait.sh
```

手动 2 卡启动：

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. \
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  train_atas.py \
  --config configs/atas_vitb_imagenet_full_author_quickgelu_2gpu_b72_allgather.yaml \
  --data-root /path/to/imagenet/train
```

从 checkpoint 恢复时增加 `--resume`：

```bash
--resume outputs/atas_vitb_imagenet_full_author_quickgelu_2gpu_b72_allgather/checkpoint_epoch_5.pt
```

## 评估

训练完成后可直接使用最终自动评估脚本：

```bash
bash scripts/run_full_author_quickgelu_2gpu_b72_allgather_post_eval.sh
```

也可以手动调用核心评估脚本。Vanilla patch matching：

```bash
python scripts/evaluate_voc_zero_shot_seg.py \
  --config configs/atas_vitb_imagenet_full_author_quickgelu_2gpu_b72_allgather.yaml \
  --voc-root /path/to/VOCdevkit/VOC2012 \
  --checkpoint /path/to/checkpoint_epoch_6.pt \
  --dense-mode vanilla \
  --output-dir outputs/voc_final_vanilla
```

SCLIP 风格评估：

```bash
python scripts/evaluate_voc_zero_shot_seg.py \
  --config configs/atas_vitb_imagenet_full_author_quickgelu_2gpu_b72_allgather.yaml \
  --voc-root /path/to/VOCdevkit/VOC2012 \
  --checkpoint /path/to/checkpoint_epoch_6.pt \
  --dense-mode sclip \
  --output-dir outputs/voc_final_sclip
```

### Checkpoint 表征漂移诊断

该脚本用于比较 ATAS checkpoint 相比冻结 CLIP teacher 的 CLS token、patch token 和 pairwise 相似度漂移，辅助判断训练是否破坏了原始 CLIP 语义：

```bash
python scripts/diagnose_checkpoint_drift.py \
  --config configs/atas_vitb_imagenet_full_author_quickgelu_2gpu_b72_allgather.yaml \
  --data-root /path/to/imagenet/train \
  --checkpoint epoch6=/path/to/checkpoint_epoch_6.pt \
  --output-dir outputs/checkpoint_drift_final
```

## 实验结果

### 作者参数 QuickGELU 对齐实验

最新完整训练配置：

```text
configs/atas_vitb_imagenet_full_author_quickgelu_2gpu_accum2.yaml
```

该配置使用 OpenAI CLIP ViT-B/16 权重并显式启用 QuickGELU。训练使用 2 卡 DDP + `gradient_accumulation_steps=2`，保持等效优化 batch 为 `144`，其余主要训练参数保持作者设置：ImageNet 全量、6 epochs、`lr=1e-5`、`weight_decay=0.1`、`GLD=1`、`LLD=0.01`、`GGD=1`、`6x6` mosaic。

| 设置 | 模型 | Foreground mIoU | Pixel Acc | Mean Class Acc |
| --- | --- | ---: | ---: | ---: |
| Vanilla | QuickGELU CLIP baseline | 0.3604 | 0.5191 | 0.5111 |
| Vanilla | QuickGELU ATAS epoch 6 | 0.3111 | 0.4729 | 0.5240 |
| Vanilla | QuickGELU ATAS b72 all-gather epoch 6 | 0.3090 | 0.4699 | 0.5198 |
| SCLIP 风格 | QuickGELU CLIP baseline | 0.7650 | 0.8602 | 0.8790 |
| SCLIP 风格 | QuickGELU ATAS epoch 6 | 0.5703 | 0.7024 | 0.7750 |
| SCLIP 风格 | QuickGELU ATAS b72 all-gather epoch 6 | 0.5817 | 0.7011 | 0.7803 |

表征漂移诊断显示，QuickGELU ATAS epoch 6 的 `mosaic_patch_cos_to_teacher_mean=-0.0393`，说明 student 的 mosaic patch token 与 teacher 几乎失去正相关。详细审计见 [作者参数 QuickGELU 复现实验审计](docs/作者参数QuickGELU复现实验审计.md)。

### All-Gather Probe

针对多卡训练中 InfoNCE 只使用本地负样本的问题，当前代码新增了：

```yaml
training:
  gather_distributed_negatives: true
```

开启后，GLD/GGD 会在 DDP 进程间 `all_gather` teacher features，把负样本扩展到全局 batch。ImageNet-100x200 子集 probe 结果：

| 设置 | 训练步数 | Vanilla mIoU | SCLIP mIoU | kNN Top-1 | Mosaic patch cosine | Patch pairwise MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QuickGELU all-gather | 80 | 0.3615 | 0.6901 | 0.9083 | 0.0595 | 0.3762 |
| QuickGELU semantic guard + all-gather | 160 | 0.3522 | 0.7410 | 0.9113 | 0.4667 | 0.0257 |

子集 probe 说明：跨 GPU 负样本和更强全局语义约束可以显著缓解 patch token 漂移。更接近作者全局 batch 的完整 ImageNet 实验已经完成：

```text
configs/atas_vitb_imagenet_full_author_quickgelu_2gpu_b72_allgather.yaml
```

该配置使用 2 GPU、每卡 batch 72、all-gather 负样本，使单步全局负样本数和等效优化 batch 都为 144。完整训练结果为 Vanilla mIoU `0.3090`、SCLIP mIoU `0.5817`，仍显著低于 QuickGELU baseline 的 `0.7650`。表征诊断中 `mosaic_patch_cos_to_teacher_mean=-0.0513`、`mosaic_patch_pairwise_mse=0.4838`，说明作者参数下 patch token 漂移仍未解决。

### 完整 ImageNet checkpoint 的 VOC2012 评估

| 设置 | 模型 | Foreground mIoU | Pixel Acc | Mean Class Acc |
| --- | --- | ---: | ---: | ---: |
| Vanilla | OpenCLIP baseline | 0.4016 | 0.5565 | 0.5680 |
| Vanilla | ATAS epoch 1 | 0.3187 | 0.4770 | 0.5364 |
| Vanilla | ATAS epoch 6 | 0.3029 | 0.4672 | 0.5152 |
| SCLIP 风格 | OpenCLIP baseline | 0.6826 | 0.8034 | 0.8292 |
| SCLIP 风格 | ATAS epoch 1 | 0.4410 | 0.5731 | 0.6435 |
| SCLIP 风格 | ATAS epoch 6 | 0.4244 | 0.5618 | 0.6388 |

更完整的实验记录见 [完整 ImageNet VOC 评估结果](docs/完整ImageNet_VOC评估结果.md)。

### Semantic Guard 子集 Probe

过程中曾验证 semantic guard probe：在 ImageNet-100x200 子集上训练 160 steps。相比完整 ATAS epoch 6，它显著降低 patch token 漂移，并把 VOC2012 结果从 `0.3029/0.4244` 回升到：

| 设置 | Foreground mIoU | Pixel Acc | Mean Class Acc |
| --- | ---: | ---: | ---: |
| Vanilla | 0.3800 | 0.5329 | 0.5665 |
| SCLIP 风格 | 0.6614 | 0.7772 | 0.8104 |

该 probe 仍未超过 OpenCLIP baseline，但说明后续改进应优先控制 patch 表征漂移。

## 当前复现差距

论文报告 ATAS 训练后 VOC20 等下游任务有提升。本仓库当前在已实现的 VOC2012 vanilla 和 SCLIP 风格评估中，ATAS checkpoint 尚未超过 OpenCLIP baseline。

可能原因包括：

- ATAS 损失实现与论文完整训练细节仍有差异。
- teacher/student token 对齐和局部区域匹配细节可能不完全一致。
- mosaic 采样策略与作者实现可能存在差别。
- OpenCLIP 与论文使用的 CLIP 实现存在差异。
- 当前下游评估是轻量复现版本，不是作者完整 dense prediction pipeline。

## 文档

- [课程作业汇报](docs/课程作业汇报.md)
- [服务器运行指南](docs/服务器运行指南.md)
- [完整 ImageNet VOC 评估结果](docs/完整ImageNet_VOC评估结果.md)
- [ATAS 三项蒸馏实现说明](docs/ATAS三项蒸馏实现说明.md)
- [作者参数 QuickGELU 复现实验审计](docs/作者参数QuickGELU复现实验审计.md)
- [消融实验方案](docs/消融实验方案.md)

## 引用

如果使用本复现项目，请优先引用原论文：

```bibtex
@inproceedings{yeo2025atas,
  title={ATAS: Any-to-Any Self-Distillation for Enhanced Open-Vocabulary Dense Prediction},
  author={Yeo, et al.},
  booktitle={ICCV},
  year={2025}
}
```

## 许可证

本仓库仅用于课程复现和学术研究。重新分发或复用前，请同时检查原论文、OpenCLIP、ImageNet 和 VOC2012 的许可证要求。
