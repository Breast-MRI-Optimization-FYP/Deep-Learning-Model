from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from kspace_transformer.data.datasets import (
    KSpaceCollator,
    TestKSpaceDataset,
    TrainKSpaceDataset,
)
from kspace_transformer.data.grids import build_lr_positions, build_normalized_grid
from kspace_transformer.data.sequence import SequenceLimiter, TruncationStats
from kspace_transformer.data.tokenize import tokenize_kspace_slice


def _write_dummy_data(tmp_path: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(0)
    hr = rng.normal(size=(4, 8, 8, 2)).astype(np.float32)
    lr = rng.normal(size=(4, 4, 4, 2)).astype(np.float32)

    mask = np.zeros((3, 8, 8), dtype=np.uint8)
    mask[:, ::2, :] = 1
    mask[:, :, ::3] = 1

    hr_path = tmp_path / "hr.npy"
    lr_path = tmp_path / "lr.npy"
    mask_path = tmp_path / "mask.npy"
    np.save(hr_path, hr)
    np.save(lr_path, lr)
    np.save(mask_path, mask)
    return hr_path, lr_path, mask_path


def test_grid_builders_return_expected_shapes() -> None:
    grid = build_normalized_grid((8, 8))
    pos, pos_norm = build_lr_positions((4, 4))

    assert grid.shape == (8, 8, 2)
    assert pos.shape == (16, 2)
    assert pos_norm.shape == (16, 2)
    assert pos.dtype == torch.int64


def test_tokenize_respects_sampled_and_unsampled_masks() -> None:
    k_gt_slice = torch.randn(8, 8, 2)
    mask = np.zeros((8, 8, 2), dtype=bool)
    mask[::2, :, :] = True
    hr_grid = build_normalized_grid((8, 8))

    sample = tokenize_kspace_slice(k_gt_slice, mask, hr_grid)

    assert sample.sampled_pos.shape[0] + sample.unsampled_pos.shape[0] == 64
    unsampled_mask = sample.selected_mask[:, :, 0].to(torch.bool)
    assert torch.all(sample.k_us[:, :, 0][unsampled_mask] == 0)


def test_sequence_limiter_updates_stats() -> None:
    k_gt_slice = torch.randn(8, 8, 2)
    mask = np.ones((8, 8, 2), dtype=bool)
    hr_grid = build_normalized_grid((8, 8))

    sample = tokenize_kspace_slice(k_gt_slice, mask, hr_grid)
    limiter = SequenceLimiter(max_seq_len=10)
    stats = TruncationStats()
    truncated = limiter.truncate(sample, stats=stats)

    assert truncated.sampled_k.shape[0] == 10
    assert stats.total_samples == 1
    assert stats.truncated_samples == 1


def test_train_dataset_and_collator(tmp_path: Path) -> None:
    hr_path, lr_path, mask_path = _write_dummy_data(tmp_path)

    train_ds = TrainKSpaceDataset(
        hr_data_path=hr_path,
        lr_data_path=lr_path,
        mask_path=mask_path,
        seed=7,
        max_seq_len=50,
    )

    item = train_ds[0]
    assert "LR_i_gt" in item
    assert "LR_k_gt" in item
    assert item["selected_mask"].shape == (8, 8, 2)

    collator = KSpaceCollator(max_seq_len=12)
    batch = collator([train_ds[0], train_ds[1]])

    assert batch["sampled_k"].shape[0] == 2
    assert batch["sampled_k"].shape[1] <= 12
    assert batch["LR_pos_norm"].shape == (2, 16, 2)


def test_test_dataset_excludes_lr_targets(tmp_path: Path) -> None:
    hr_path, _, mask_path = _write_dummy_data(tmp_path)

    test_ds = TestKSpaceDataset(
        hr_data_path=hr_path,
        mask_path=mask_path,
        lr_resolution=(4, 4),
        seed=3,
    )
    item = test_ds[0]

    assert "LR_i_gt" not in item
    assert "LR_k_gt" not in item
    assert item["LR_pos_norm"].shape == (16, 2)
