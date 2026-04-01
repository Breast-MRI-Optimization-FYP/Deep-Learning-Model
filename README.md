Implementation of a Transformer model operating within the k-space domain for breast MRI optimization. The repository was built incrementally following a modular approach, with each subsystem designed as an independent component that can be tested, profiled, and improved in isolation.

## 1. Project Objective

The primary objective is breast MRI optimization through high-fidelity reconstruction from undersampled k-space data. The system targets stable improvements in reconstruction quality (PSNR, SSIM) while maintaining practical runtime and memory behavior.

Research goals:

- improve breast MRI reconstruction quality from limited k-space samples,
- preserve physically consistent k-space/image-domain relationships,
- support reproducible staged training and evaluation,
- provide measurable acceptance criteria for candidate model updates.

## 2. Incremental Modular Build Strategy

The repository is organized as modular research software. Data processing, model design, training control, inference, validation, and CLI orchestration are separated into dedicated modules. This structure supports:

- rapid experimentation at subsystem boundaries,
- unit and integration testing by module,
- transparent benchmarking and parity checks,
- easier adaptation to alternative breast MRI datasets and mask regimes.

## 3. Model Architecture

### 3.1 Input/Output Representation

- Complex-valued tensors are represented with two channels: `[..., 2]` for real/imaginary parts.
- Frequency/image transforms use centered FFT/IFFT utilities.
- Sampled and unsampled token streams are built from masked k-space coordinates.

### 3.2 Core Network Structure

The model (`KSpaceTransformer`) combines:

1. Transformer encoder over sampled k-space tokens.
2. LR decoder predicting low-resolution image-space outputs.
3. HR decoder predicting unsampled high-resolution k-space/image outputs.
4. CNN refinement blocks with data consistency in the RM stage.

The architecture includes positional encoding, multi-head attention, feed-forward transformer blocks, and stage-aware HR refinement.

### 3.3 Stage-Wise Learning Flow

Training progresses through three stages:

- `LR`: low-resolution reconstruction learning,
- `K`: high-resolution transformer prediction without refinement,
- `RM`: transformer + CNN refinement with data consistency.

Stage scheduling controls active losses, convolutional refinement weight, and evaluation interval per stage.

## 4. Performance-Oriented Features

Implemented features that improve reconstruction quality, stability, or efficiency include:

- stage-aware weighted loss computation for LR/HR/RM outputs,
- strict runtime configuration and shape/data contract validation,
- deterministic seeding for reproducibility,
- adaptive mask reassignment during training,
- sequence-length control for tokenized sampled/unsampled streams,
- data-consistency enforcement in refinement blocks,
- AdamW optimization with cosine learning-rate schedule,
- checkpoint lifecycle (`last` and best-by-PSNR),
- TensorBoard metric logging with collision-safe metric keys,
- runtime and peak-memory telemetry in workflow summaries,
- parity gate comparing baseline vs candidate runs with metric and resource thresholds.

## 5. Repository Architecture

```text
.
├── cli/          # Executable workflows: train/test/preprocess/split/parity
├── config/       # Typed runtime schema, defaults, CLI argument binding
├── data/         # Tokenization, masks, grids, datasets, LR generation, split pipeline
├── inference/    # Inference runner and stage-aware evaluation path
├── model/        # Transformer, attention, decoders, refinement blocks
├── training/     # Trainer engine, stage scheduler, losses, metrics, checkpoints
├── utils/        # FFT utilities, device helpers, seed control, runtime tracker
├── validation/   # Parity comparison and acceptance gate logic
├── tests/        # Unit/integration/smoke coverage
├── pyproject.toml
└── README.md
```

## 6. Data Expectations

### 6.1 Core Arrays

- HR k-space array: shape `[N, H, W, 2]`, `float32`
- LR k-space array: shape `[N, h, w, 2]`, `float32`
- Mask bank: shape `[M, H, W]` or `[M, H, W, 2]`

### 6.2 Split Outputs

The split workflow writes:

- `train_k.npy`, `valid_k.npy`, `test_k.npy`
- `train_lr_k.npy`, `valid_lr_k.npy`, `test_lr_k.npy`
- `split_indices.npz`

## 7. Requirements

- Python `>=3.10,<3.14`
- NumPy `>=2.1,<3.0`
- PyTorch `>=2.2,<3.0`
- h5py `>=3.10,<4.0`
- scikit-image `>=0.22,<1.0`
- TensorBoard `>=2.15,<3.0`
- tqdm `>=4.66,<5.0`

Optional development tools:

- pytest, pytest-cov, ruff, mypy

## 8. Setup

### 8.1 Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 8.2 Installation

Runtime install:

```bash
python -m pip install -e .
```

Development install:

```bash
python -m pip install -e ".[dev]"
```

## 9. Command Line Workflows

Entrypoints:

- `kst-preprocess`
- `kst-split`
- `kst-train`
- `kst-test`
- `kst-parity`

### 9.1 Generate LR K-Space from HR K-Space

```bash
kst-preprocess \
	--input_hr_kspace_path ./data/processed/train_k.npy \
	--output_lr_kspace_path ./data/processed/train_lr_k.npy \
	--scale 2 \
	--batch_size 50 \
	--save_summary_path ./runs/preprocess_summary.json
```

### 9.2 Deterministic Train/Validation/Test Split

```bash
kst-split \
	--hr_kspace_path ./data/processed/processed_data.npy \
	--lr_kspace_path ./data/processed/LR_k_data.npy \
	--split_output_dir ./data/processed/splits \
	--train_ratio 0.7 \
	--valid_ratio 0.15 \
	--test_ratio 0.15 \
	--shuffle true \
	--random_seed 42 \
	--save_summary_path ./runs/split_summary.json
```

### 9.3 Train

```bash
kst-train \
	--output_dir ./runs/exp01 \
	--train_hr_data_path ./data/processed/splits/train_k.npy \
	--train_lr_data_path ./data/processed/splits/train_lr_k.npy \
	--train_mask_path ./data/masks/combined_masks.npy \
	--valid_hr_data_path ./data/processed/splits/valid_k.npy \
	--valid_lr_data_path ./data/processed/splits/valid_lr_k.npy \
	--valid_mask_path ./data/masks/combined_masks.npy \
	--epoch_num 200 \
	--pure_lr_training_epoch 50 \
	--pure_k_training_epoch 100
```

Primary artifacts under `--output_dir`:

- `tensorboard/`
- `checkpoints/last.pth`
- `checkpoints/best_valid_psnr.pth`
- `training_summary.json`

### 9.4 Inference/Evaluation

```bash
kst-test \
	--output_dir ./runs/exp01 \
	--checkpoint ./runs/exp01/checkpoints/best_valid_psnr.pth \
	--test_hr_data_path ./data/processed/splits/test_k.npy \
	--test_mask_path ./data/masks/combined_masks.npy \
	--inference_stage RM \
	--save_summary_path ./runs/exp01/inference_summary.json
```

### 9.5 Parity Gate (Baseline vs Candidate)

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

Exit status semantics:

- `0`: parity gate pass
- `1`: parity gate fail

## 10. Reproducibility and Validation

- Deterministic seed configuration is enabled through runtime options.
- Metric reporting includes PSNR and SSIM for training and evaluation.
- Runtime summaries provide elapsed time and peak memory signals.
- Test suite includes contracts, data modules, model forward behavior, engine integration, CLI smoke coverage, and parity validation.

Run tests:

```bash
python -m pytest -q
```

## 11. Notes for Breast MRI Experiments

- Ensure mask banks are aligned with HR spatial resolution.
- Keep acquisition-specific preprocessing steps consistent across train/validation/test splits.
- Use parity reports when introducing architectural or hyperparameter changes to preserve clinical-quality reconstruction behavior.
