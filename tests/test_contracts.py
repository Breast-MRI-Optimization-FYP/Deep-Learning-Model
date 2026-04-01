from __future__ import annotations

import pytest
import torch

from data.types import BatchTensors, ForwardOutputs, LossBreakdown, TokenizedSample


def test_tokenized_sample_validation_passes() -> None:
    sample = TokenizedSample(
        sampled_k=torch.randn(10, 2),
        sampled_pos=torch.randint(0, 64, (10, 2), dtype=torch.int64),
        sampled_pos_norm=torch.randn(10, 2),
        unsampled_pos=torch.randint(0, 64, (20, 2), dtype=torch.int64),
        unsampled_pos_norm=torch.randn(20, 2),
        k_us=torch.randn(8, 8, 2),
        selected_mask=torch.ones(8, 8, 2, dtype=torch.bool),
    )
    sample.validate()


def test_tokenized_sample_invalid_shape_fails() -> None:
    sample = TokenizedSample(
        sampled_k=torch.randn(10, 3),
        sampled_pos=torch.randint(0, 64, (10, 2), dtype=torch.int64),
        sampled_pos_norm=torch.randn(10, 2),
        unsampled_pos=torch.randint(0, 64, (20, 2), dtype=torch.int64),
        unsampled_pos_norm=torch.randn(20, 2),
        k_us=torch.randn(8, 8, 2),
        selected_mask=torch.ones(8, 8, 2, dtype=torch.bool),
    )
    with pytest.raises(ValueError):
        sample.validate()


def test_batch_tensors_validation_passes() -> None:
    batch = BatchTensors(
        sampled_k=torch.randn(2, 12, 2),
        sampled_pos_norm=torch.randn(2, 12, 2),
        unsampled_pos=torch.randint(0, 64, (2, 20, 2), dtype=torch.int64),
        unsampled_pos_norm=torch.randn(2, 20, 2),
        k_us=torch.randn(2, 8, 8, 2),
        i_gt=torch.randn(2, 8, 8, 2),
        k_gt=torch.randn(2, 8, 8, 2),
        selected_mask=torch.ones(2, 8, 8, 2, dtype=torch.bool),
        lr_i_gt=torch.randn(2, 4, 4, 2),
        lr_k_gt=torch.randn(2, 4, 4, 2),
        lr_pos_norm=torch.randn(2, 16, 2),
    )
    batch.validate()


def test_forward_outputs_batch_mismatch_fails() -> None:
    outputs = ForwardOutputs(
        lr_images=[torch.randn(2, 4, 4, 2)],
        up_lr_image=torch.randn(3, 8, 8, 2),
    )
    with pytest.raises(ValueError):
        outputs.validate()


def test_loss_breakdown_length_validation() -> None:
    breakdown = LossBreakdown(
        total_loss=torch.tensor(1.0),
        lr_img_losses=[0.1, 0.2],
        lr_k_losses=[0.2, 0.1],
        hr_img_losses=[0.1, 0.1, 0.1],
        hr_k_losses=[0.2, 0.2, 0.2],
        rm_img_losses=[0.3, 0.3, 0.3],
        rm_k_losses=[0.4, 0.4, 0.4],
    )
    with pytest.raises(ValueError):
        breakdown.validate(expected_lr_layers=2, expected_hr_layers=4)

