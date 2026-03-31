from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .types import SplitIndices


def _save_split(data_mmap: np.ndarray, indices: np.ndarray, output_path: Path, batch_size: int = 100) -> None:
    output_shape = (len(indices), *data_mmap.shape[1:])
    output_mmap = np.lib.format.open_memmap(
        str(output_path),
        mode="w+",
        dtype=data_mmap.dtype,
        shape=output_shape,
    )

    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        output_mmap[start:end] = data_mmap[indices[start:end]]

    del output_mmap


def split_data_memory_efficient(
    k_data_path: str | Path,
    lr_k_data_path: str | Path,
    *,
    output_dir: str | Path,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True,
    random_seed: int = 42,
) -> SplitIndices:
    ratio_sum = train_ratio + valid_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + valid_ratio + test_ratio must sum to 1.0")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k_data_mmap = np.load(str(k_data_path), mmap_mode="r")
    lr_data_mmap = np.load(str(lr_k_data_path), mmap_mode="r")
    if k_data_mmap.shape[0] != lr_data_mmap.shape[0]:
        raise ValueError("HR and LR sample counts must match")

    total_samples = int(k_data_mmap.shape[0])
    indices = np.arange(total_samples)

    if shuffle:
        rng = np.random.default_rng(random_seed)
        rng.shuffle(indices)

    train_end = int(total_samples * train_ratio)
    valid_end = train_end + int(total_samples * valid_ratio)

    train_indices = indices[:train_end]
    valid_indices = indices[train_end:valid_end]
    test_indices = indices[valid_end:]

    _save_split(k_data_mmap, train_indices, output_dir / "train_k.npy")
    _save_split(k_data_mmap, valid_indices, output_dir / "valid_k.npy")
    _save_split(k_data_mmap, test_indices, output_dir / "test_k.npy")

    _save_split(lr_data_mmap, train_indices, output_dir / "train_lr_k.npy")
    _save_split(lr_data_mmap, valid_indices, output_dir / "valid_lr_k.npy")
    _save_split(lr_data_mmap, test_indices, output_dir / "test_lr_k.npy")

    np.savez(
        output_dir / "split_indices.npz",
        train=train_indices,
        valid=valid_indices,
        test=test_indices,
        random_seed=random_seed,
    )

    split_indices = SplitIndices(
        train=torch.from_numpy(train_indices.astype(np.int64)),
        valid=torch.from_numpy(valid_indices.astype(np.int64)),
        test=torch.from_numpy(test_indices.astype(np.int64)),
    )
    split_indices.validate()
    return split_indices
