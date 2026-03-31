from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kspace_transformer.config.cli import add_test_args, parse_runtime_config
from kspace_transformer.data.datasets import KSpaceCollator, TestKSpaceDataset
from kspace_transformer.inference.engine import InferenceRunner
from kspace_transformer.model import KSpaceTransformer
from kspace_transformer.training.checkpoint import CheckpointManager
from kspace_transformer.training.logger import RunLogger
from kspace_transformer.training.stage import TrainingStage
from kspace_transformer.utils.device import resolve_device
from kspace_transformer.utils.seed import set_global_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference for K-Space Transformer")
    add_test_args(parser)
    return parser


def _build_model_from_config(config) -> KSpaceTransformer:
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


def main_test(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Parse-time config validation uses train weights lengths; align synthetic defaults with selected layer counts.
    if not hasattr(args, "lr_weights") or getattr(args, "lr_weights", None) is None:
        setattr(args, "lr_weights", [1.0] * int(args.num_LRdecoder_layers))
    if not hasattr(args, "hr_weights") or getattr(args, "hr_weights", None) is None:
        setattr(args, "hr_weights", [1.0] * int(args.num_HRdecoder_layers))

    config = parse_runtime_config(args)

    run_logger = RunLogger(prefix="test")
    run_logger.info("Runtime configuration validated.")

    checkpoint_path = _require_existing(config.paths.checkpoint, "checkpoint")
    test_hr_path = _require_existing(config.paths.test_hr_data_path, "test_hr_data_path")
    test_mask_path = _require_existing(config.paths.test_mask_path, "test_mask_path")

    num_workers = int(getattr(args, "num_workers", 0))
    if num_workers < 0:
        raise ValueError("--num_workers must be >= 0")

    set_global_seed(config.runtime.seed, deterministic=True)

    collator = KSpaceCollator(max_seq_len=config.data.max_seq_len)
    dataset = TestKSpaceDataset(
        hr_data_path=test_hr_path,
        mask_path=test_mask_path,
        lr_resolution=(config.data.lr_size, config.data.lr_size),
        seed=config.runtime.seed,
        max_seq_len=config.data.max_seq_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )

    model = _build_model_from_config(config)
    device = resolve_device(config.runtime.gpu)

    checkpoint_manager = CheckpointManager(Path(config.paths.output_dir) / "checkpoints")
    checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        missing_keys, unexpected_keys = checkpoint_manager.load_model_state(
            model, checkpoint, strict=False
        )
    else:
        incompatible = model.load_state_dict(checkpoint, strict=False)
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)

    if missing_keys or unexpected_keys:
        run_logger.info(
            f"Loaded checkpoint with relaxed compatibility: missing={len(missing_keys)}, "
            f"unexpected={len(unexpected_keys)}"
        )

    stage = TrainingStage[str(getattr(args, "inference_stage", "RM")).upper()]
    runner = InferenceRunner(model=model, device=device)
    summary = runner.run_test_set(
        loader,
        stage=stage,
        up_scale=int(getattr(args, "up_scale", 2)),
        conv_weight=config.train.conv_weight,
    )

    summary_payload = {
        "checkpoint": str(checkpoint_path),
        "test_hr_data_path": str(test_hr_path),
        "test_mask_path": str(test_mask_path),
        "stage": summary["stage"],
        "mean_psnr": summary["mean_psnr"],
        "mean_ssim": summary["mean_ssim"],
        "num_samples": summary["num_samples"],
    }

    summary_path_arg = getattr(args, "save_summary_path", None)
    if summary_path_arg:
        summary_path = Path(summary_path_arg)
    else:
        summary_path = Path(config.paths.output_dir) / "inference_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    run_logger.info(f"Inference summary saved to {summary_path}")
    run_logger.info(
        f"stage={summary_payload['stage']} mean_psnr={summary_payload['mean_psnr']:.4f} "
        f"mean_ssim={summary_payload['mean_ssim']:.6f} samples={summary_payload['num_samples']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_test())
