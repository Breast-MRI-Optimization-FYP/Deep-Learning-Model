from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config.cli import add_train_args, parse_runtime_config
from data.datasets import KSpaceCollator, TrainKSpaceDataset, ValidKSpaceDataset
from model import KSpaceTransformer
from training.checkpoint import CheckpointManager
from training.engine import Trainer
from training.logger import RunLogger, TensorboardLogger
from training.losses import LossComputer
from training.stage import StageScheduler
from utils.device import resolve_device
from utils.perf import RuntimeTracker
from utils.seed import set_global_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train K-Space Transformer model")
    add_train_args(parser)
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


def _build_lr_scheduler(optimizer: AdamW, epoch_num: int) -> LambdaLR:
    total_epochs = max(int(epoch_num), 1)

    def _lambda(epoch: int) -> float:
        return 0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))

    return LambdaLR(optimizer, lr_lambda=_lambda)


def main_train(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = parse_runtime_config(args, require_paths=True)

    run_logger = RunLogger(prefix="train")
    run_logger.info("Runtime configuration validated.")
    run_logger.info(
        "Stage boundaries: "
        f"LR<= {config.train.pure_lr_training_epoch}, "
        f"K<= {config.train.pure_k_training_epoch}, "
        "RM> pure_k_training_epoch"
    )

    num_workers = int(getattr(args, "num_workers", 0))
    if num_workers < 0:
        raise ValueError("--num_workers must be >= 0")

    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(config.runtime.seed, deterministic=True)
    device = resolve_device(config.runtime.gpu)
    runtime_tracker = RuntimeTracker(device=device)
    runtime_tracker.start()

    collator = KSpaceCollator(max_seq_len=config.data.max_seq_len)
    train_dataset = TrainKSpaceDataset(
        hr_data_path=config.paths.train_hr_data_path,
        lr_data_path=config.paths.train_lr_data_path,
        mask_path=config.paths.train_mask_path,
        seed=config.runtime.seed,
        max_seq_len=config.data.max_seq_len,
    )
    valid_dataset = ValidKSpaceDataset(
        hr_data_path=config.paths.valid_hr_data_path,
        lr_data_path=config.paths.valid_lr_data_path,
        mask_path=config.paths.valid_mask_path,
        seed=config.runtime.seed + 1,
        max_seq_len=config.data.max_seq_len,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.data.valid_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
    )

    model = _build_model_from_config(config)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        run_logger.info(f"Using DataParallel across {torch.cuda.device_count()} CUDA devices")
        model = torch.nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.l2norm)
    lr_scheduler = _build_lr_scheduler(optimizer, config.train.epoch_num)

    stage_scheduler = StageScheduler(
        pure_lr_training_epoch=config.train.pure_lr_training_epoch,
        pure_k_training_epoch=config.train.pure_k_training_epoch,
        lr_weights=config.train.lr_weights,
        hr_weights=config.train.hr_weights,
        conv_weight=config.train.conv_weight,
        num_lr_layers=config.model.num_lrdecoder_layers,
        num_hr_layers=config.model.num_hrdecoder_layers,
        eval_interval_lr=config.eval.eval_interval_lr,
        eval_interval_k=config.eval.eval_interval_k,
        eval_interval_rm=config.eval.eval_interval_rm,
    )

    loss_computer = LossComputer(
        kspace_loss=config.train.kspace_loss,
        img_loss=config.train.img_loss,
    )

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    metric_logger = TensorboardLogger(writer)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        stage_scheduler=stage_scheduler,
        loss_computer=loss_computer,
        device=device,
        lr_scheduler=lr_scheduler,
        metric_logger=metric_logger,
    )

    checkpoint_manager = CheckpointManager(output_dir / "checkpoints")
    start_epoch = 1
    best_valid_psnr = float("-inf")
    best_valid_ssim = float("-inf")

    if config.runtime.resume_train:
        if config.paths.checkpoint is None:
            raise ValueError("--resume_train requires --checkpoint path")
        checkpoint = checkpoint_manager.load_checkpoint(config.paths.checkpoint, map_location=device)
        target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        checkpoint_manager.load_model_state(target_model, checkpoint, strict=False)

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "lr_sch_state_dict" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_sch_state_dict"])

        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_valid_psnr = float(checkpoint.get("best_valid_psnr", best_valid_psnr))
        best_valid_ssim = float(checkpoint.get("best_valid_ssim", best_valid_ssim))
        if best_valid_psnr != float("-inf"):
            checkpoint_manager.best_metric = best_valid_psnr
        run_logger.info(f"Resumed from checkpoint {config.paths.checkpoint} at epoch {start_epoch}")

    if start_epoch > config.train.epoch_num:
        run_logger.info("Start epoch exceeds configured epoch_num; nothing to train.")
        writer.close()
        return 0

    history: list[dict[str, float | int | str | None]] = []
    for epoch in range(start_epoch, config.train.epoch_num + 1):
        if config.data.reassign_mask_every > 0 and epoch % config.data.reassign_mask_every == 0:
            train_dataset.reassign_mask()

        train_result = trainer.train_epoch(train_loader, epoch=epoch)
        stage_weights = stage_scheduler.weights_for_epoch(epoch)

        valid_result = None
        if epoch == start_epoch or epoch % stage_weights.eval_interval == 0:
            valid_result = trainer.validate_epoch(valid_loader, epoch=epoch)

        trainer.log_epoch_metrics(train_result, valid_result)

        if valid_result is not None:
            best_valid_psnr = max(best_valid_psnr, valid_result.psnr)
            best_valid_ssim = max(best_valid_ssim, valid_result.ssim)

        serializable_state = {
            "epoch": epoch,
            "model_state_dict": (model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_sch_state_dict": lr_scheduler.state_dict(),
            "best_valid_psnr": best_valid_psnr,
            "best_valid_ssim": best_valid_ssim,
            "stage": train_result.stage.value,
            "train_metrics": {
                "loss": train_result.loss,
                "psnr": train_result.psnr,
                "ssim": train_result.ssim,
            },
            "valid_metrics": (
                {
                    "loss": valid_result.loss,
                    "psnr": valid_result.psnr,
                    "ssim": valid_result.ssim,
                }
                if valid_result is not None
                else None
            ),
        }
        checkpoint_manager.save_checkpoint(serializable_state, filename="last.pth")

        improved = False
        if valid_result is not None:
            improved, _ = checkpoint_manager.save_best(
                serializable_state,
                metric=valid_result.psnr,
                mode="max",
                filename="best_valid_psnr.pth",
            )

        run_logger.info(
            f"Epoch {epoch}/{config.train.epoch_num} stage={train_result.stage.value} "
            f"train_psnr={train_result.psnr:.4f} "
            f"valid_psnr={(valid_result.psnr if valid_result is not None else float('nan')):.4f} "
            f"best_updated={improved}"
        )

        history.append(
            {
                "epoch": epoch,
                "stage": train_result.stage.value,
                "train_loss": train_result.loss,
                "train_psnr": train_result.psnr,
                "train_ssim": train_result.ssim,
                "valid_loss": valid_result.loss if valid_result is not None else None,
                "valid_psnr": valid_result.psnr if valid_result is not None else None,
                "valid_ssim": valid_result.ssim if valid_result is not None else None,
            }
        )

        if config.runtime.test_first_epoch and epoch == 1:
            run_logger.info("test_first_epoch enabled; stopping after first epoch")
            break

    writer.close()

    summary = {
        "output_dir": str(output_dir),
        "start_epoch": start_epoch,
        "end_epoch": history[-1]["epoch"] if history else start_epoch - 1,
        "best_valid_psnr": best_valid_psnr if math.isfinite(best_valid_psnr) else None,
        "best_valid_ssim": best_valid_ssim if math.isfinite(best_valid_ssim) else None,
        "runtime_seconds": runtime_tracker.elapsed_seconds(),
        "peak_memory_bytes": runtime_tracker.peak_memory_bytes(),
        "history": history,
    }
    summary_path_arg = getattr(args, "save_summary_path", None)
    if summary_path_arg:
        summary_path = Path(summary_path_arg)
    else:
        summary_path = output_dir / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    run_logger.info(f"Training summary saved to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main_train())

