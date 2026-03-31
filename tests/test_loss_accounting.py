from __future__ import annotations

import pytest
import torch

from kspace_transformer.training.losses import LossComputer
from kspace_transformer.training.stage import TrainingStage


def test_lr_loss_vectors_are_independent() -> None:
    computer = LossComputer(kspace_loss=True, img_loss=True)

    preds = [torch.zeros(1, 4, 4, 2), torch.ones(1, 4, 4, 2)]
    lr_i_gt = torch.zeros(1, 4, 4, 2)
    lr_k_gt = torch.zeros(1, 4, 4, 2)

    breakdown = computer.compute_lr_losses(
        lr_predictions=preds,
        lr_i_gt=lr_i_gt,
        lr_k_gt=lr_k_gt,
        lr_weights=(0.5, 0.5),
        to_kspace=lambda tensor: 2.0 * tensor,
    )

    assert len(breakdown.lr_img_losses) == 2
    assert len(breakdown.lr_k_losses) == 2
    assert breakdown.lr_img_losses[1] == pytest.approx(1.0)
    assert breakdown.lr_k_losses[1] == pytest.approx(4.0)
    assert breakdown.lr_img_losses[1] != breakdown.lr_k_losses[1]


def test_hr_k_stage_does_not_emit_rm_losses() -> None:
    computer = LossComputer(kspace_loss=True, img_loss=True)

    transformer_preds = [torch.ones(1, 4, 4, 2)]
    refined_preds = [torch.ones(1, 4, 4, 2) * 3]
    target = torch.zeros(1, 4, 4, 2)

    breakdown = computer.compute_hr_losses(
        hr_transformer_predictions=transformer_preds,
        hr_refined_predictions=refined_preds,
        i_gt=target,
        k_gt=target,
        hr_weights=(1.0,),
        conv_weight=1.0,
        stage=TrainingStage.K,
        to_kspace=lambda tensor: tensor,
    )

    assert breakdown.hr_img_losses[0] > 0
    assert breakdown.hr_k_losses[0] > 0
    assert breakdown.rm_img_losses[0] == pytest.approx(0.0)
    assert breakdown.rm_k_losses[0] == pytest.approx(0.0)


def test_hr_rm_stage_emits_refinement_losses() -> None:
    computer = LossComputer(kspace_loss=True, img_loss=True)

    transformer_preds = [torch.ones(1, 4, 4, 2)]
    refined_preds = [torch.ones(1, 4, 4, 2) * 2]
    target = torch.zeros(1, 4, 4, 2)

    breakdown = computer.compute_hr_losses(
        hr_transformer_predictions=transformer_preds,
        hr_refined_predictions=refined_preds,
        i_gt=target,
        k_gt=target,
        hr_weights=(1.0,),
        conv_weight=0.5,
        stage=TrainingStage.RM,
        to_kspace=lambda tensor: tensor,
    )

    assert breakdown.hr_img_losses[0] > 0
    assert breakdown.hr_k_losses[0] > 0
    assert breakdown.rm_img_losses[0] > 0
    assert breakdown.rm_k_losses[0] > 0
    assert float(breakdown.total_loss.item()) > 0
