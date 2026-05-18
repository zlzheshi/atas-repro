# VOC checkpoint sweep 与 MaskCLIP 尝试记录

## 实验目的

当前 2 万张 ImageNet 子集训练得到的 ATAS checkpoint 在 kNN 上有小幅提升，但在 VOC2012 的直接 patch matching 评估上没有超过原始 OpenCLIP。为了判断问题是“某个 epoch 过训练”还是“训练规模与下游框架不够接近作者设置”，我们对 epoch 1 到 epoch 6 做了一轮完整 VOC2012 sweep。

同时，我们尝试在现有 OpenCLIP 代码路径上加入一个近似的 MaskCLIP value-token 提取模式，用来评估是否能更接近论文里的 MaskCLIP 下游设置。这个尝试目前只能作为失败记录，不能作为正式 MaskCLIP 复现结果。

## VOC checkpoint sweep

运行脚本：

```bash
GPU=3 bash scripts/run_voc_checkpoint_sweep.sh
```

输出目录：

```text
outputs/voc_checkpoint_sweep_subset_100x200/
```

汇总结果：

| 模型 | 前景 mIoU | 前景像素准确率 | 平均类别准确率 |
| --- | ---: | ---: | ---: |
| OpenCLIP baseline | 0.4016 | 0.5565 | 0.5680 |
| ATAS epoch 1 | 0.3731 | 0.5342 | 0.5930 |
| ATAS epoch 2 | 0.3671 | 0.5242 | 0.5858 |
| ATAS epoch 3 | 0.3604 | 0.5174 | 0.5789 |
| ATAS epoch 4 | 0.3358 | 0.4896 | 0.5494 |
| ATAS epoch 5 | 0.3334 | 0.4865 | 0.5470 |
| ATAS epoch 6 | 0.3287 | 0.4854 | 0.5462 |

结论：

- 子集 checkpoint 在 VOC2012 直接 patch matching 上没有超过原始 OpenCLIP。
- epoch 1 是子集 ATAS 中的最佳 VOC dense proxy checkpoint，但 mIoU 仍低于 baseline。
- epoch 越往后，ImageNet 子集 kNN 虽然更好，VOC dense proxy 却持续下降，说明当前 2 万张子集训练不足以支撑论文中的 dense prediction 增益。
- 这组结果支持下一步转向完整 ImageNet 和更接近作者的下游评估框架，而不是继续延长子集训练。

## 近似 MaskCLIP 尝试

运行脚本：

```bash
GPU=3 bash scripts/run_voc_maskclip_eval.sh
```

输出目录：

```text
outputs/voc_maskclip_subset_100x200/
```

汇总结果：

| 模型 | 前景 mIoU | 前景像素准确率 | 平均类别准确率 |
| --- | ---: | ---: | ---: |
| OpenCLIP baseline, 近似 MaskCLIP | 0.0159 | 0.1418 | 0.0586 |
| ATAS epoch 1, 近似 MaskCLIP | 0.0160 | 0.1409 | 0.0533 |
| ATAS epoch 3, 近似 MaskCLIP | 0.0161 | 0.1390 | 0.0520 |
| ATAS epoch 6, 近似 MaskCLIP | 0.0163 | 0.1354 | 0.0505 |

结论：

- 这个结果明显异常，mIoU 远低于 vanilla patch matching。
- 原因很可能是当前实现只粗略抽取最后一层 attention value 分支，没有完整复现 MaskCLIP 的特征重组、归一化和推理细节。
- 因此它不能用于和论文 Table 1 的 MaskCLIP 结果比较，只能作为“简单近似不可用”的排错记录。
- 后续如果继续做 MaskCLIP/SCLIP，应优先接入官方或社区成熟实现，而不是在当前脚本里继续堆近似逻辑。

## 对后续路线的影响

作者论文中 VOC20 Vanilla CLIP 从 41.8 mIoU 提升到 ATAS 的 56.0 mIoU。我们当前 OpenCLIP baseline 在 VOC2012 直接 patch matching 上为 40.16 mIoU，和论文 baseline 比较接近，说明数据与基础评估方向大体合理。差距主要来自两个方面：

1. 训练数据规模：当前子集只有 20,000 张图，论文使用完整 ImageNet 约 120 万张图。
2. 下游框架：论文主结果包含 Vanilla CLIP、MaskCLIP、SCLIP、ClearCLIP 等成熟 dense prediction 推理框架，当前只有直接 patch matching 和一个失败的近似 MaskCLIP 尝试。

因此最优先的路线已经调整为：完整 ImageNet 作者设置训练 -> 等 checkpoint 产生 -> 用 vanilla VOC2012 先评估 -> 再接入更可靠的 MaskCLIP/SCLIP 评估。
