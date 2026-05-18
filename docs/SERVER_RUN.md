# Server Run Guide

This guide assumes the lab server is Linux with NVIDIA GPUs.

The current A6000 server setup uses:

- Project directory: `/mnt/t1b6/xuzhejia/atas-repro`
- Conda root: `/mnt/t1b6/xuzhejia/app/miniconda3`
- Conda environment: `atas`

## 1. Upload Code

From your local machine, upload the repository:

```bash
scp -r atas-repro user@server:/home/user/projects/
```

Or use GitHub after creating the course repository:

```bash
git clone <your-repo-url>
cd atas-repro
```

## 2. Create Environment

```bash
conda create -n atas python=3.10 -y
conda activate atas
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Check the environment:

```bash
python scripts/check_env.py
nvidia-smi
```

## 3. Prepare Data

ATAS trains on unlabeled object-centric images. The closest reproduction target is ImageNet train in ImageFolder format:

```text
/data/imagenet/train/
  n01440764/
    xxx.JPEG
  n01443537/
    yyy.JPEG
```

Labels are ignored by the training script.

If ImageNet is not available yet, use a small ImageFolder-style subset first to test the pipeline.

## 4. Debug Run

Always run the debug config first:

```bash
python train_atas.py \
  --config configs/atas_vitb_debug.yaml \
  --data-root /data/imagenet/train
```

Expected result: one epoch starts, losses print, and a checkpoint appears under `outputs/atas_vitb_debug/`.

## 5. Formal Training

Single GPU:

```bash
bash scripts/run_train_atas.sh configs/atas_vitb_24gb.yaml /data/imagenet/train
```

Multi GPU:

```bash
GPUS=4 bash scripts/run_train_atas.sh configs/atas_vitb.yaml /data/imagenet/train
```

On the configured A6000 server, use the path-aware helper:

```bash
cd /mnt/t1b6/xuzhejia/atas-repro
CUDA_VISIBLE_DEVICES=0 GPUS=1 bash scripts/a6000_run.sh \
  configs/atas_vitb_debug.yaml \
  /mnt/t1b6/xuzhejia/datasets/imagenet/train
```

For four GPUs after confirming they are idle:

```bash
cd /mnt/t1b6/xuzhejia/atas-repro
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 bash scripts/a6000_run.sh \
  configs/atas_vitb.yaml \
  /mnt/t1b6/xuzhejia/datasets/imagenet/train
```

Resume:

```bash
python train_atas.py \
  --config configs/atas_vitb_24gb.yaml \
  --data-root /data/imagenet/train \
  --resume outputs/atas_vitb_24gb/checkpoint_epoch_3.pt
```

## 6. Recommended Settings

- 4 GPUs with 24GB or more: use `configs/atas_vitb.yaml`.
- 1 GPU with 24GB: use `configs/atas_vitb_24gb.yaml`.
- Less than 24GB: reduce `data.image_size` to `576` or `384`, reduce `training.batch_size`, and increase `gradient_accumulation_steps`.

## 7. What to Record

For the course report, record:

- GPU type and count.
- Dataset size.
- Effective batch size: `batch_size * GPUs * gradient_accumulation_steps`.
- Training loss curves for GLD, LLD, GGD.
- Checkpoint path used for downstream evaluation.
