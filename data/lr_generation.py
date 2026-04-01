from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from utils.fftc import fft2c, ifft2c

from .preprocess import down_sample_i


def _compute_lr_shape(sample_hr: np.ndarray, scale: int) -> tuple[int, int, int]:
    sample_t = torch.from_numpy(sample_hr[None, ...].astype(np.float32))
    sample_i = ifft2c(sample_t, need_shift=True).cpu().numpy().astype(np.float32)
    sample_lr_i = down_sample_i(sample_i, scale=scale)
    return tuple(sample_lr_i.shape[1:])


def generate_lr_kspace_batches(
    input_path: str | Path,
    output_path: str | Path,
    *,
    scale: int = 2,
    batch_size: int = 50,
) -> Path:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    k_data_mmap = np.load(str(input_path), mmap_mode="r")
    if k_data_mmap.ndim != 4 or k_data_mmap.shape[-1] != 2:
        raise ValueError("input data must have shape [N,H,W,2]")

    output_hwc = _compute_lr_shape(k_data_mmap[0], scale=scale)
    output_shape = (k_data_mmap.shape[0], *output_hwc)

    lr_k_data_mmap = np.lib.format.open_memmap(
        str(output_path),
        mode="w+",
        dtype=np.float32,
        shape=output_shape,
    )

    total_samples = int(k_data_mmap.shape[0])
    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)

        k_batch_np = k_data_mmap[start:end].astype(np.float32, copy=True)
        k_batch_t = torch.from_numpy(k_batch_np)

        i_batch = ifft2c(k_batch_t, need_shift=True).cpu().numpy().astype(np.float32)
        lr_i_batch = down_sample_i(i_batch, scale=scale)
        lr_k_batch = fft2c(torch.from_numpy(lr_i_batch)).cpu().numpy().astype(np.float32)

        lr_k_data_mmap[start:end] = lr_k_batch

    del lr_k_data_mmap
    return output_path

