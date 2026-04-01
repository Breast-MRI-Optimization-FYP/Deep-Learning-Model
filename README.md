# Deep-Learning-Model

Ground up built implementation for the K-Space Transformer MRI reconstruction project.

## Current Status

This repository is ground up built for the source problem domain.
The active objective is correctness-first development.

## Scope Decisions

- Stage scheduling follows LR -> K -> RM training progression.
- Data payload keys preserve source compatibility.

## Setup

Install in editable mode:

```bash
python -m pip install -e .
```

Run tests:

```bash
python -m pytest -q
```

## Entrypoints

- `kst-train`
- `kst-test`
- `kst-preprocess`
- `kst-split`
- `kst-parity`

## Command Usage

### 1) Generate LR k-space from HR k-space

```bash
kst-preprocess \
	--input_hr_kspace_path ./data/processed/train_k.npy \
	--output_lr_kspace_path ./data/processed/train_lr_k.npy \
	--scale 2 \
	--batch_size 50
```

### 2) Split HR and LR datasets

```bash
kst-split \
	--hr_kspace_path ./data/processed/processed_data.npy \
	--lr_kspace_path ./data/processed/LR_k_data.npy \
	--split_output_dir ./data/processed/splits \
	--train_ratio 0.7 \
	--valid_ratio 0.15 \
	--test_ratio 0.15 \
	--shuffle true \
	--random_seed 42
```

### 3) Train

```bash
kst-train \
	--output_dir ./runs/exp01 \
	--train_hr_data_path ./data/processed/splits/train_k.npy \
	--train_lr_data_path ./data/processed/splits/train_lr_k.npy \
	--train_mask_path ./data/masks/combined_masks.npy \
	--valid_hr_data_path ./data/processed/splits/valid_k.npy \
	--valid_lr_data_path ./data/processed/splits/valid_lr_k.npy \
	--valid_mask_path ./data/masks/combined_masks.npy
```

Training artifacts are written under `--output_dir`, including:

- `tensorboard/`
- `checkpoints/last.pth`
- `checkpoints/best_valid_psnr.pth`
- `training_summary.json`

### 4) Inference

```bash
kst-test \
	--output_dir ./runs/exp01 \
	--checkpoint ./runs/exp01/checkpoints/best_valid_psnr.pth \
	--test_hr_data_path ./data/processed/splits/test_k.npy \
	--test_mask_path ./data/masks/combined_masks.npy \
	--inference_stage RM \
	--save_summary_path ./runs/exp01/inference_summary.json
```

### 5) Parity Gate (Baseline vs Candidate)

```bash
kst-parity \
	--baseline_train_summary ./runs/baseline/training_summary.json \
	--baseline_inference_summary ./runs/baseline/inference_summary.json \
	--candidate_train_summary ./runs/candidate/training_summary.json \
	--candidate_inference_summary ./runs/candidate/inference_summary.json \
	--psnr_drift_db 0.05 \
	--ssim_drift 0.001 \
	--runtime_drift_ratio 0.05 \
	--memory_drift_ratio 0.10 \
	--output_report ./runs/candidate/parity_report.json
```

## Notes

- Masks are interpreted as sampled indicators at selection time, and converted to unsampled indicators for model-facing data-consistency operations.
- Model and CLI are designed for parity-first validation, not architecture redesign.
