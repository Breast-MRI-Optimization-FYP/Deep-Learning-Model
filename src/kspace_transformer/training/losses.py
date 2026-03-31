from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn

from kspace_transformer.data.types import LossBreakdown
from kspace_transformer.training.stage import TrainingStage

TensorTransform = Callable[[torch.Tensor], torch.Tensor]


class LossComputer:
    def __init__(
        self,
        *,
        kspace_loss: bool,
        img_loss: bool,
        loss_fn: nn.Module | None = None,
    ) -> None:
        self.kspace_loss = kspace_loss
        self.img_loss = img_loss
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()

    def compute_lr_losses(
        self,
        *,
        lr_predictions: Sequence[torch.Tensor],
        lr_i_gt: torch.Tensor,
        lr_k_gt: torch.Tensor,
        lr_weights: Sequence[float],
        to_kspace: TensorTransform,
    ) -> LossBreakdown:
        if len(lr_predictions) != len(lr_weights):
            raise ValueError("lr_predictions length must equal lr_weights length")

        total_loss = torch.zeros((), device=lr_i_gt.device)
        lr_img_losses: list[float] = []
        lr_k_losses: list[float] = []

        for layer_idx, prediction in enumerate(lr_predictions):
            layer_weight = float(lr_weights[layer_idx])
            img_value = 0.0
            k_value = 0.0

            if self.kspace_loss:
                k_loss_tensor = self.loss_fn(to_kspace(prediction), lr_k_gt)
                total_loss = total_loss + layer_weight * k_loss_tensor
                k_value = float(k_loss_tensor.detach().item())

            if self.img_loss:
                img_loss_tensor = self.loss_fn(prediction, lr_i_gt)
                total_loss = total_loss + layer_weight * img_loss_tensor
                img_value = float(img_loss_tensor.detach().item())

            # Keep independent vectors to avoid cross-assignment between image and k-space losses.
            lr_img_losses.append(img_value)
            lr_k_losses.append(k_value)

        breakdown = LossBreakdown(
            total_loss=total_loss,
            lr_img_losses=lr_img_losses,
            lr_k_losses=lr_k_losses,
            hr_img_losses=[],
            hr_k_losses=[],
            rm_img_losses=[],
            rm_k_losses=[],
        )
        breakdown.validate(expected_lr_layers=len(lr_predictions))
        return breakdown

    def compute_hr_losses(
        self,
        *,
        hr_transformer_predictions: Sequence[torch.Tensor],
        hr_refined_predictions: Sequence[torch.Tensor],
        i_gt: torch.Tensor,
        k_gt: torch.Tensor,
        hr_weights: Sequence[float],
        conv_weight: float,
        stage: TrainingStage,
        to_kspace: TensorTransform,
    ) -> LossBreakdown:
        if len(hr_transformer_predictions) != len(hr_refined_predictions):
            raise ValueError(
                "hr_transformer_predictions length must equal hr_refined_predictions length"
            )
        if len(hr_transformer_predictions) != len(hr_weights):
            raise ValueError("hr prediction length must equal hr_weights length")

        if stage not in {TrainingStage.K, TrainingStage.RM}:
            raise ValueError("compute_hr_losses expects stage to be K or RM")

        total_loss = torch.zeros((), device=i_gt.device)
        hr_img_losses: list[float] = []
        hr_k_losses: list[float] = []
        rm_img_losses: list[float] = []
        rm_k_losses: list[float] = []

        for layer_idx, (transformer_pred, refined_pred) in enumerate(
            zip(hr_transformer_predictions, hr_refined_predictions)
        ):
            layer_weight = float(hr_weights[layer_idx])

            hr_img_value = 0.0
            hr_k_value = 0.0
            rm_img_value = 0.0
            rm_k_value = 0.0

            if self.kspace_loss:
                hr_k_tensor = self.loss_fn(to_kspace(transformer_pred), k_gt)
                total_loss = total_loss + layer_weight * hr_k_tensor
                hr_k_value = float(hr_k_tensor.detach().item())

                if stage is TrainingStage.RM and conv_weight > 0:
                    rm_k_tensor = self.loss_fn(to_kspace(refined_pred), k_gt)
                    total_loss = total_loss + conv_weight * layer_weight * rm_k_tensor
                    rm_k_value = float(rm_k_tensor.detach().item())

            if self.img_loss:
                hr_img_tensor = self.loss_fn(transformer_pred, i_gt)
                total_loss = total_loss + layer_weight * hr_img_tensor
                hr_img_value = float(hr_img_tensor.detach().item())

                if stage is TrainingStage.RM and conv_weight > 0:
                    rm_img_tensor = self.loss_fn(refined_pred, i_gt)
                    total_loss = total_loss + conv_weight * layer_weight * rm_img_tensor
                    rm_img_value = float(rm_img_tensor.detach().item())

            hr_img_losses.append(hr_img_value)
            hr_k_losses.append(hr_k_value)
            rm_img_losses.append(rm_img_value)
            rm_k_losses.append(rm_k_value)

        breakdown = LossBreakdown(
            total_loss=total_loss,
            lr_img_losses=[],
            lr_k_losses=[],
            hr_img_losses=hr_img_losses,
            hr_k_losses=hr_k_losses,
            rm_img_losses=rm_img_losses,
            rm_k_losses=rm_k_losses,
        )
        breakdown.validate(expected_hr_layers=len(hr_transformer_predictions))
        return breakdown
