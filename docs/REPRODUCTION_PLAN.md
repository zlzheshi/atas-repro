# ATAS Reproduction Plan

## Goal

Reproduce the core method of ATAS: improve CLIP ViT patch representations for open-vocabulary dense prediction through self-distillation on unlabeled object-centric images.

## Minimum Viable Reproduction

The minimum version should include:

1. ATAS training on ImageNet or an ImageNet-style object-centric subset.
2. Three distillation losses:
   - GLD: global-to-local distillation.
   - LLD: local-to-local relation distillation.
   - GGD: global-to-global distillation.
3. Mosaic augmentation with object-centric images.
4. Loss ablation:
   - CLIP baseline.
   - GLD only.
   - GLD + LLD.
   - GLD + LLD + GGD.
5. Zero-shot segmentation evaluation on at least one public dataset.

## Recommended Course Scope

Priority 1: Backbone training

- Use OpenAI CLIP ViT-B/16 through OpenCLIP.
- Train only the visual encoder.
- Use the original CLIP visual encoder as the frozen teacher.
- Use ImageNet train if available.

Priority 2: Lightweight dense evaluation

- Start from Pascal VOC semantic segmentation.
- Evaluate patch-level class prediction using CLIP text embeddings.
- Report mIoU and mAcc if masks are available.

Priority 3: Paper-style segmentation

- Add MaskCLIP or SCLIP feature extraction.
- Compare original CLIP and ATAS checkpoint.
- Try VOC20, Pascal Context 59, and COCO-Stuff if data is available.

Priority 4: Optional detection extension

- Use F-ViT or a compatible open-vocabulary detector.
- Replace CLIP visual weights with the ATAS checkpoint.
- Evaluate OV-COCO novel AP50.

## Experiments to Report

Main experiment:

| Model | Training Data | Segmentation mIoU | Notes |
| --- | --- | ---: | --- |
| CLIP | none | TBD | original backbone |
| ATAS | object-centric subset | TBD | our reproduction |

Ablation:

| GLD | LLD | GGD | mIoU | Observation |
| --- | --- | --- | ---: | --- |
| yes | no | no | TBD | alignment only |
| yes | yes | no | TBD | local relation preserved |
| yes | yes | yes | TBD | full ATAS |

Mosaic:

| Mosaic | mIoU | Observation |
| --- | ---: | --- |
| no mosaic | TBD | object-centric only |
| 2x2 | TBD | small composite |
| 2x2 + 4x4 + 6x6 | TBD | closest to paper |

## Risk Control

- If full ImageNet is unavailable, use ImageNet-100 or a balanced subset and state the limitation.
- If 960 resolution is too expensive, train at 384 or 576 first and report it as a course-scale reproduction.
- If MaskCLIP/SCLIP integration is delayed, use patch-level zero-shot segmentation as the first measurable evaluation.
- Detection should be optional because F-ViT integration is much heavier than the ATAS backbone training itself.

## Division of Labor

- Training engineer: maintain `train_atas.py`, configs, checkpoints.
- Data engineer: prepare ImageNet/VOC/Context/COCO-Stuff paths.
- Evaluation engineer: implement segmentation metrics and visualization.
- Ablation engineer: run loss and mosaic ablations.
- Documentation engineer: record environment, commands, results, and failed attempts.

