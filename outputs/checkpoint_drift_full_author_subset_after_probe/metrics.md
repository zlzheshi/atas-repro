# Checkpoint Drift Diagnostics

| name | cls_cos_to_teacher_mean | global_patch_cos_to_teacher_mean | mosaic_patch_cos_to_teacher_mean | cls_pairwise_mse | mosaic_patch_pairwise_mse | loss_gld | loss_lld | loss_ggd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 3.591748 | 0.000000 | 3.411303 |
| full_epoch1 | 0.662690 | 0.660973 | -0.036658 | 0.057344 | 0.565909 | 3.410519 | 0.566153 | 3.342771 |
| full_epoch6 | 0.648413 | 0.515938 | -0.043033 | 0.056106 | 0.576401 | 3.348660 | 0.576112 | 3.345998 |
| semantic_guard_probe | 0.690455 | 0.955477 | 0.663271 | 0.061185 | 0.010220 | 3.586560 | 0.010136 | 3.335283 |
