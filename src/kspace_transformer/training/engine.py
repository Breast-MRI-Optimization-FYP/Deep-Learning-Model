from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.optim import Optimizer

from kspace_transformer.data.types import ForwardOutputs
from kspace_transformer.utils.device import to_device
from kspace_transformer.utils.fftc import fft2c

from .logger import TensorboardLogger
from .losses import LossComputer
from .metrics import MetricsAccumulator, compute_psnr, compute_ssim
from .stage import StageScheduler, TrainingStage


@dataclass(slots=True)
class EpochResult:
    epoch: int
    stage: TrainingStage
    loss: float
    psnr: float
    ssim: float


class Trainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: Optimizer,
        stage_scheduler: StageScheduler,
        loss_computer: LossComputer,
        device: torch.device,
        lr_scheduler: Any | None = None,
        metric_logger: TensorboardLogger | None = None,
        up_scale: int = 2,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.stage_scheduler = stage_scheduler
        self.loss_computer = loss_computer
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.metric_logger = metric_logger
        self.up_scale = up_scale

        self.model.to(self.device)

    def _normalize_outputs(self, raw_outputs: Any) -> ForwardOutputs:
        if isinstance(raw_outputs, ForwardOutputs):
            raw_outputs.validate()
            return raw_outputs

        if isinstance(raw_outputs, tuple) and len(raw_outputs) == 5:
            lr_images, up_lr_image, up_lr_k, hr_images, hr_refined_images = raw_outputs
            outputs = ForwardOutputs(
                lr_images=list(lr_images),
                up_lr_image=up_lr_image,
                up_lr_k=up_lr_k,
                hr_transformer_images=list(hr_images),
                hr_refined_images=list(hr_refined_images),
            )
            outputs.validate()
            return outputs

        raise TypeError("Model output must be ForwardOutputs or tuple with length 5")

    def _run_forward(self, batch: dict[str, torch.Tensor], stage: TrainingStage, conv_weight: float) -> ForwardOutputs:
        raw = self.model(
            src=batch["sampled_k"],
            lr_pos=batch["LR_pos_norm"],
            src_pos=batch["sampled_pos_norm"],
            hr_pos=batch["unsampled_pos_norm"],
            k_us=batch["k_us"],
            unsampled_pos=batch["unsampled_pos"],
            up_scale=self.up_scale,
            mask=batch["selected_mask"],
            conv_weight=conv_weight,
            stage=stage.value,
        )
        return self._normalize_outputs(raw)

    def _select_prediction_for_metrics(
        self,
        outputs: ForwardOutputs,
        stage: TrainingStage,
    ) -> torch.Tensor:
        if stage is TrainingStage.LR:
            if outputs.up_lr_image is not None:
                return outputs.up_lr_image
            return outputs.lr_images[-1]

        if stage is TrainingStage.K:
            return outputs.hr_transformer_images[-1]

        return outputs.hr_refined_images[-1]

    def _compute_batch_loss(
        self,
        batch: dict[str, torch.Tensor],
        outputs: ForwardOutputs,
        *,
        stage: TrainingStage,
        conv_weight: float,
        lr_weights: tuple[float, ...],
        hr_weights: tuple[float, ...],
    ) -> torch.Tensor:
        if "LR_i_gt" not in batch or "LR_k_gt" not in batch:
            raise KeyError("Training batch must include LR_i_gt and LR_k_gt")

        lr_breakdown = self.loss_computer.compute_lr_losses(
            lr_predictions=outputs.lr_images,
            lr_i_gt=batch["LR_i_gt"],
            lr_k_gt=batch["LR_k_gt"],
            lr_weights=lr_weights,
            to_kspace=fft2c,
        )
        total_loss = lr_breakdown.total_loss

        if stage is not TrainingStage.LR:
            hr_breakdown = self.loss_computer.compute_hr_losses(
                hr_transformer_predictions=outputs.hr_transformer_images,
                hr_refined_predictions=outputs.hr_refined_images,
                i_gt=batch["i_gt"],
                k_gt=batch["k_gt"],
                hr_weights=hr_weights,
                conv_weight=conv_weight,
                stage=stage,
                to_kspace=fft2c,
            )
            total_loss = total_loss + hr_breakdown.total_loss

        return total_loss

    def _run_epoch(self, loader: Any, *, epoch: int, train: bool) -> EpochResult:
        stage_weights = self.stage_scheduler.weights_for_epoch(epoch)
        if train:
            self.model.train()
        else:
            self.model.eval()

        acc = MetricsAccumulator()
        context = torch.enable_grad() if train else torch.no_grad()

        with context:
            for batch in loader:
                batch = to_device(batch, self.device)
                outputs = self._run_forward(
                    batch,
                    stage=stage_weights.stage,
                    conv_weight=stage_weights.conv_weight,
                )
                total_loss = self._compute_batch_loss(
                    batch,
                    outputs,
                    stage=stage_weights.stage,
                    conv_weight=stage_weights.conv_weight,
                    lr_weights=stage_weights.lr_weights,
                    hr_weights=stage_weights.hr_weights,
                )

                if train:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                acc.update("loss", float(total_loss.detach().item()))

                pred_for_metrics = self._select_prediction_for_metrics(outputs, stage_weights.stage)
                i_gt = batch["i_gt"]
                batch_size = int(i_gt.shape[0])
                psnr_sum = compute_psnr(pred_for_metrics.detach(), i_gt.detach())
                ssim_sum = compute_ssim(pred_for_metrics.detach(), i_gt.detach())
                acc.update("psnr", float(psnr_sum), n=batch_size)
                acc.update("ssim", float(ssim_sum), n=batch_size)

        if train and self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return EpochResult(
            epoch=epoch,
            stage=stage_weights.stage,
            loss=acc.mean("loss"),
            psnr=acc.mean("psnr"),
            ssim=acc.mean("ssim"),
        )

    def train_epoch(self, loader: Any, *, epoch: int) -> EpochResult:
        return self._run_epoch(loader, epoch=epoch, train=True)

    def validate_epoch(self, loader: Any, *, epoch: int) -> EpochResult:
        return self._run_epoch(loader, epoch=epoch, train=False)

    def _log_epoch_metrics(self, train_result: EpochResult, valid_result: EpochResult | None) -> None:
        if self.metric_logger is None:
            return

        for metric_name, value in {
            "loss": train_result.loss,
            "psnr": train_result.psnr,
            "ssim": train_result.ssim,
        }.items():
            self.metric_logger.log_metric(
                split="train",
                stage=train_result.stage.value,
                family="metrics",
                metric=metric_name,
                value=value,
                step=train_result.epoch,
            )

        if valid_result is None:
            return

        for metric_name, value in {
            "loss": valid_result.loss,
            "psnr": valid_result.psnr,
            "ssim": valid_result.ssim,
        }.items():
            self.metric_logger.log_metric(
                split="valid",
                stage=valid_result.stage.value,
                family="metrics",
                metric=metric_name,
                value=value,
                step=valid_result.epoch,
            )

    def log_epoch_metrics(self, train_result: EpochResult, valid_result: EpochResult | None) -> None:
        self._log_epoch_metrics(train_result, valid_result)

    def fit(
        self,
        *,
        train_loader: Any,
        valid_loader: Any | None,
        start_epoch: int,
        end_epoch: int,
    ) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []

        for epoch in range(start_epoch, end_epoch + 1):
            train_result = self.train_epoch(train_loader, epoch=epoch)
            stage_weights = self.stage_scheduler.weights_for_epoch(epoch)

            valid_result: EpochResult | None = None
            should_validate = (
                valid_loader is not None
                and (epoch == start_epoch or epoch % stage_weights.eval_interval == 0)
            )
            if should_validate:
                valid_result = self.validate_epoch(valid_loader, epoch=epoch)

            self._log_epoch_metrics(train_result, valid_result)
            history.append(
                {
                    "epoch": epoch,
                    "stage": train_result.stage,
                    "train": train_result,
                    "valid": valid_result,
                }
            )

        return history
