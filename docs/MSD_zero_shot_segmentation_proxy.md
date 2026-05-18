# MSD Zero-shot Segmentation Proxy

## 实验目的

为了让 ATAS 复现更接近论文的 open-vocabulary dense prediction 目标，本轮在服务器上加入了一个轻量级 patch-text 零样本分割代理评估。

由于官方 VOC2012 下载速度极慢，当前先使用服务器上已有的 MSD 数据集做 dense proxy。MSD 是手机屏幕表面缺陷分割数据集，格式接近 PASCAL VOC，包含 3 类前景缺陷：Oil、Stain、Scratch。

这个实验不是论文标准 benchmark 的完全替代，但可以回答一个关键问题：ATAS 训练后的 ViT patch token 是否更容易与文本类别对齐，从而支持密集预测。

## 数据与设置

- 数据集：`/data/haier/wangcairui/dataset/MSD`
- split：`val`
- 验证图像数：240
- 类别映射：`1=oil stain`，`2=stain`，`3=scratch`
- 输入分辨率：448
- Backbone：OpenCLIP ViT-B/16
- ATAS checkpoint：`outputs/atas_vitb_subset_100x200_stable/checkpoint_epoch_3.pt`
- 预测方式：提取 ViT patch token，与 CLIP text embedding 做 cosine 相似度，再上采样到像素级。

运行命令：

```bash
GPU=3 OUTPUT_SUFFIX=_balanced SAVE_VIS=0 SAVE_VIS_PER_CLASS=4 bash scripts/run_msd_zero_shot_seg.sh
GPU=3 BG_CLASS='clean phone screen background' SAVE_VIS=0 SAVE_VIS_PER_CLASS=4 bash scripts/run_msd_zero_shot_seg.sh
```

## 前景三分类结果

该设置只在前景像素上判断 `oil stain / stain / scratch`，不引入 background 类。它更接近“patch 前景类别判别能力”评估。

| Model | foreground mIoU | foreground pixel acc | mean class acc | Oil IoU | Stain IoU | Scratch IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenCLIP baseline | 0.0940 | 0.2059 | 0.3143 | 0.1809 | 0.0338 | 0.0674 |
| Full ATAS epoch 3 | 0.2575 | 0.7428 | 0.3124 | 0.7454 | 0.0018 | 0.0252 |

结论：

- Full ATAS 的 foreground mIoU 从 0.0940 提升到 0.2575，foreground pixel acc 从 0.2059 提升到 0.7428。
- 提升主要来自 Oil 类，说明 ATAS patch token 更倾向于形成强前景响应，但在 Stain/Scratch 上还不稳定。
- mean class acc 基本没有提升，说明这个结果受 MSD 类别/像素不均衡影响明显，不能只看 pixel acc。

输出目录：

```text
outputs/msd_zero_shot_seg_baseline_balanced/
outputs/msd_zero_shot_seg_full_atas_balanced/
```

可视化格式为：原图、GT、prediction、overlay。颜色：红色=oil stain，蓝色=stain，橙色=scratch。

示例：

```text
outputs/msd_zero_shot_seg_full_atas_balanced/visualizations/Oil_0013_seg.png
outputs/msd_zero_shot_seg_full_atas_balanced/visualizations/Scr_0002_seg.png
outputs/msd_zero_shot_seg_full_atas_balanced/visualizations/Sta_0001_seg.png
```

## 加入 Background 的严格分割结果

该设置增加文本类 `clean phone screen background`，同时评估背景与前景。它更接近完整 semantic segmentation，但也更依赖 background prompt 的设计。

| Model | mIoU | pixel acc | foreground mIoU | foreground pixel acc | Background IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenCLIP baseline | 0.1623 | 0.6470 | 0.0006 | 0.0524 | 0.6474 |
| Full ATAS epoch 3 | 0.1255 | 0.4989 | 0.0014 | 0.2598 | 0.4980 |

结论：

- 加入 background 后，两个模型的 foreground mIoU 都很低，说明简单 patch-text matching 对背景建模很敏感。
- Full ATAS 的 foreground pixel acc 高于 baseline，但 background IoU 下降，整体 mIoU 反而低于 baseline。
- 因此，这组结果应作为“严格分割代理实验的限制分析”，不宜作为 ATAS 优于 baseline 的主证据。

输出目录：

```text
outputs/msd_zero_shot_seg_baseline_with_bg/
outputs/msd_zero_shot_seg_full_atas_with_bg/
```

## 与论文复现的关系

目前已经形成一条比较完整的课程复现实验证据链：

1. ImageNet 子集训练：复现 GLD/LLD/GGD 自蒸馏训练流程。
2. kNN 评估：Full ATAS 保持全局语义能力，同时 centroid margin 显著提升。
3. 消融实验：GLD/LLD 单独使用会破坏全局语义，GGD 是稳定训练的关键。
4. patch alignment 可视化：展示局部 patch 响应变化。
5. MSD zero-shot segmentation proxy：补充 dense prediction 风格的定量与可视化结果。

后续如果官方 VOC2012 下载完成，可以直接运行 `scripts/run_voc_zero_shot_seg.sh`，得到更接近论文 benchmark 的零样本语义分割结果。
