from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.cli import add_test_args, parse_runtime_config
from data.datasets import KSpaceCollator
from evaluation.acceleration import build_mask_metadata
from evaluation.baselines import load_baseline_metrics
from evaluation.dataset import EvaluationKSpaceDataset
from evaluation.qualitative import save_qualitative_samples
from evaluation.reporting import write_evaluation_reports
from evaluation.runner import EvaluationRunner
from evaluation.schemas import EvaluationConfig
from model import KSpaceTransformer
from training.checkpoint import CheckpointManager
from training.logger import RunLogger
from utils.device import resolve_device
from utils.seed import set_global_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Comprehensive K-Space Transformer evaluation")
    add_test_args(parser)
    parser.add_argument("--mask_manifest", type=str, default=None)
    parser.add_argument("--acceleration_factors", nargs="*", type=float, default=[])
    parser.add_argument("--evaluation_stages", nargs="+", choices=["K", "RM"], default=["K", "RM"])
    parser.add_argument("--qualitative_samples_per_acceleration", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--baseline_metrics", type=str, default=None)
    return parser


def _build_model(config) -> KSpaceTransformer:
    return KSpaceTransformer(
        lr_size=config.data.lr_size,
        channel=2,
        d_model=config.model.d_model,
        nhead=config.model.n_head,
        num_encoder_layers=config.model.num_encoder_layers,
        num_lrdecoder_layers=config.model.num_lrdecoder_layers,
        num_hrdecoder_layers=config.model.num_hrdecoder_layers,
        dim_feedforward=config.model.dim_feedforward,
        hr_conv_channel=config.model.hr_conv_channel,
        hr_conv_num=config.model.hr_conv_num,
        hr_kernel_size=config.model.hr_kernel_size,
        dropout=config.train.dropout,
        activation="relu",
    )


def _require_existing(path: Path | None, name: str) -> Path:
    if path is None:
        raise ValueError(f"{name} is required")
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def main_evaluate(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if not hasattr(args, "lr_weights") or getattr(args, "lr_weights", None) is None:
        setattr(args, "lr_weights", [1.0] * int(args.num_LRdecoder_layers))
    if not hasattr(args, "hr_weights") or getattr(args, "hr_weights", None) is None:
        setattr(args, "hr_weights", [1.0] * int(args.num_HRdecoder_layers))
    config = parse_runtime_config(args)

    logger = RunLogger(prefix="evaluate")
    checkpoint_path = _require_existing(config.paths.checkpoint, "checkpoint")
    test_hr_path = _require_existing(config.paths.test_hr_data_path, "test_hr_data_path")
    test_mask_path = _require_existing(config.paths.test_mask_path, "test_mask_path")
    output_dir = Path(config.paths.output_dir) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(args.num_workers) < 0:
        raise ValueError("--num_workers must be >= 0")
    if args.max_samples is not None and int(args.max_samples) <= 0:
        raise ValueError("--max_samples must be positive")

    set_global_seed(config.runtime.seed, deterministic=True)
    device = resolve_device(config.runtime.gpu)
    mask_bank = np.load(str(test_mask_path))
    mask_metadata = build_mask_metadata(
        mask_bank,
        manifest_path=args.mask_manifest,
        requested_accelerations=tuple(float(value) for value in args.acceleration_factors),
    )

    dataset = EvaluationKSpaceDataset(
        hr_data_path=test_hr_path,
        mask_path=test_mask_path,
        mask_metadata=mask_metadata,
        lr_resolution=(config.data.lr_size, config.data.lr_size),
        max_seq_len=config.data.max_seq_len,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=KSpaceCollator(max_seq_len=config.data.max_seq_len),
    )

    model = _build_model(config)
    checkpoint_manager = CheckpointManager(output_dir / "checkpoints")
    checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        logger.info(
            f"Relaxed checkpoint load: missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )

    evaluation_config = EvaluationConfig(
        output_dir=output_dir,
        stages=tuple(args.evaluation_stages),
        up_scale=int(args.up_scale),
        conv_weight=config.train.conv_weight,
        qualitative_samples_per_acceleration=int(args.qualitative_samples_per_acceleration),
    )
    result = EvaluationRunner(
        model=model,
        device=device,
        mask_metadata=mask_metadata,
        config=evaluation_config,
    ).run(loader)

    qualitative_paths = save_qualitative_samples(
        result.qualitative_samples,
        output_dir / "qualitative",
    )
    report_paths = write_evaluation_reports(
        result,
        output_dir,
        metadata={
            "checkpoint": str(checkpoint_path),
            "test_hr_data_path": str(test_hr_path),
            "test_mask_path": str(test_mask_path),
            "mask_manifest": args.mask_manifest,
            "stages": list(evaluation_config.stages),
            "num_test_slices": int(dataset.k_gt.shape[0]),
            "num_masks": len(mask_metadata),
            "qualitative_outputs": [str(path) for path in qualitative_paths],
        },
        baseline_rows=load_baseline_metrics(args.baseline_metrics) if args.baseline_metrics else [],
    )
    logger.info(f"Evaluation complete: {len(result.records)} metric records")
    logger.info(f"Summary saved to {report_paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_evaluate())
