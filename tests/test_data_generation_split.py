from __future__ import annotations

from pathlib import Path

import numpy as np

from data.lr_generation import generate_lr_kspace_batches
from data.preprocess import (
    apply_variance_based_slice_selection,
    down_sample_i,
    normalize_image_slice,
)
from data.split import split_data_memory_efficient


def test_generate_lr_kspace_batches_shape(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    hr = rng.normal(size=(5, 8, 8, 2)).astype(np.float32)

    input_path = tmp_path / "hr.npy"
    output_path = tmp_path / "lr_generated.npy"
    np.save(input_path, hr)

    result_path = generate_lr_kspace_batches(input_path, output_path, scale=2, batch_size=2)
    generated = np.load(result_path)

    assert generated.shape == (5, 4, 4, 2)


def test_split_data_memory_efficient_outputs(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    hr = rng.normal(size=(10, 8, 8, 2)).astype(np.float32)
    lr = rng.normal(size=(10, 4, 4, 2)).astype(np.float32)

    hr_path = tmp_path / "hr.npy"
    lr_path = tmp_path / "lr.npy"
    output_dir = tmp_path / "splits"
    np.save(hr_path, hr)
    np.save(lr_path, lr)

    split_indices = split_data_memory_efficient(
        hr_path,
        lr_path,
        output_dir=output_dir,
        train_ratio=0.7,
        valid_ratio=0.15,
        test_ratio=0.15,
        shuffle=False,
    )

    assert split_indices.train.shape[0] == 7
    assert split_indices.valid.shape[0] == 1
    assert split_indices.test.shape[0] == 2

    assert np.load(output_dir / "train_k.npy").shape[0] == 7
    assert np.load(output_dir / "valid_k.npy").shape[0] == 1
    assert np.load(output_dir / "test_k.npy").shape[0] == 2

    assert np.load(output_dir / "train_lr_k.npy").shape[0] == 7
    assert np.load(output_dir / "valid_lr_k.npy").shape[0] == 1
    assert np.load(output_dir / "test_lr_k.npy").shape[0] == 2


def test_preprocess_helpers() -> None:
    const = np.ones((8, 8, 2), dtype=np.float32)
    normalized = normalize_image_slice(const)
    assert np.allclose(normalized, 0.0)

    images = np.random.default_rng(2).normal(size=(2, 8, 8, 2)).astype(np.float32)
    down = down_sample_i(images, scale=2)
    assert down.shape == (2, 4, 4, 2)

    volume = np.random.default_rng(3).normal(size=(10, 8, 8, 2)).astype(np.float32)
    volume[0] = 0.0
    reduced, removed, kept = apply_variance_based_slice_selection(volume, removal_percentage=0.2)
    assert reduced.shape[0] == 8
    assert removed.shape[0] == 2
    assert kept.shape[0] == 8

