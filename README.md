# ATAS 复现项目

本仓库是论文 **ATAS: Any-to-Any Self-Distillation for Enhanced Open-Vocabulary Dense Prediction** 的 PyTorch 复现实现，重点覆盖 CLIP ViT 图像编码器自蒸馏训练，以及基于 PASCAL VOC2012 的开放词汇语义分割评估。

> 当前状态：已完成 ATAS 训练主流程、ImageNet 多卡训练、断点续训、VOC2012 vanilla patch matching 评估和 SCLIP 风格 dense inference 评估。当前复现实验尚未达到论文中下游指标提升的效果。

## 特性

- 基于 OpenCLIP ViT-B/16 的 teacher-student 自蒸馏训练。
- 支持 6x6 mosaic 输入构造。
- 实现 ATAS 核心损失：`GLD`、`LLD`、`GGD`。
- 支持 AMP、checkpoint 保存/恢复、DDP 多卡训练。
- 支持 VOC2012 零样本语义分割评估。
- 支持 SCLIP 风格 self-correlation dense inference。
- 保留轻量级实验结果和可复现实验配置。

## 项目结构

```text
configs/                 训练配置文件
scripts/                 数据准备、训练、评估和可视化脚本
src/                     项目公共模块
outputs/                 轻量级指标结果和部分可视化结果
docs/                    课程汇报、服务器运行指南和实验结果记录
train_atas.py            ATAS 训练入口
requirements.txt         Python 依赖
```

大型数据集、模型权重和训练 checkpoint 不提交到 git。

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

主要配置文件：

```text
configs/atas_vitb_imagenet_full_author.yaml
```

该配置使用 ViT-B/16，训练 6 个 epoch，batch size 为每卡 36，优化器为 AdamW，学习率 `1e-5`，权重衰减 `0.1`，损失权重为 `GLD=1`、`LLD=0.01`、`GGD=1`。

4 卡 DDP 训练命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. \
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  train_atas.py \
  --config configs/atas_vitb_imagenet_full_author.yaml \
  --data-root /path/to/imagenet/train
```

从 checkpoint 恢复：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. \
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  train_atas.py \
  --config configs/atas_vitb_imagenet_full_author.yaml \
  --data-root /path/to/imagenet/train \
  --resume outputs/atas_vitb_imagenet_full_author/checkpoint_epoch_5.pt
```

实验室服务器上可使用等待空闲 GPU 的启动脚本：

```bash
bash scripts/run_full_imagenet_author_wait.sh
```

## 评估

### VOC2012 Vanilla Patch Matching

```bash
GPU=0 \
VOC_ROOT=/path/to/VOCdevkit/VOC2012 \
bash scripts/run_voc_full_author_sweep.sh
```

输出目录：

```text
outputs/voc_full_author_sweep/
```

### VOC2012 SCLIP 风格评估

```bash
GPU=0 \
VOC_ROOT=/path/to/VOCdevkit/VOC2012 \
bash scripts/run_voc_sclip_eval.sh
```

输出目录：

```text
outputs/voc_sclip_full_author/
```

### ImageNet 子集 kNN 评估

```bash
python scripts/evaluate_imagenet_subset_knn.py \
  --config configs/atas_vitb_subset_100x200_stable.yaml \
  --data-root /path/to/imagenet_subset_100x200/train \
  --checkpoint outputs/atas_vitb_subset_100x200_stable/checkpoint_epoch_6.pt \
  --output-dir outputs/eval_subset_100x200_epoch6_knn
```

### Checkpoint 表征漂移诊断

该脚本用于比较 ATAS checkpoint 相比冻结 CLIP teacher 的 CLS token、patch token 和 pairwise 相似度漂移，辅助判断训练是否破坏了原始 CLIP 语义：

```bash
python scripts/diagnose_checkpoint_drift.py \
  --config configs/atas_vitb_subset_100x200_stable.yaml \
  --data-root /path/to/imagenet_subset_100x200/train \
  --checkpoint epoch1=outputs/atas_vitb_imagenet_full_author/checkpoint_epoch_1.pt \
  --checkpoint epoch6=outputs/atas_vitb_imagenet_full_author/checkpoint_epoch_6.pt \
  --output-dir outputs/checkpoint_drift_full_author_subset
```

## 实验结果

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

为减少 ATAS 训练对 CLIP 局部表征的破坏，仓库补充了一个保守消融配置：

```text
configs/atas_vitb_subset_100x200_semantic_guard_probe.yaml
```

该 probe 在 ImageNet-100x200 子集上训练 160 steps。相比完整 ATAS epoch 6，它显著降低 patch token 漂移，并把 VOC2012 结果从 `0.3029/0.4244` 回升到：

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
