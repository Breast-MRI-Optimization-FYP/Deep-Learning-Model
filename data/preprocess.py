from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def normalize_image_slice(image: np.ndarray) -> np.ndarray:
    mean = float(np.mean(image))
    std = float(np.std(image))
    if std == 0.0:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - mean) / std).astype(np.float32)


def apply_variance_based_slice_selection(
    volume: np.ndarray,
    removal_percentage: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if volume.ndim != 4:
        raise ValueError("volume must have shape [N,H,W,C]")
    if not (0.0 <= removal_percentage < 1.0):
        raise ValueError("removal_percentage must be in [0.0, 1.0)")

    num_slices = volume.shape[0]
    if num_slices == 0:
        return volume, np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if volume.shape[-1] == 2:
        magnitude = np.sqrt(volume[:, :, :, 0] ** 2 + volume[:, :, :, 1] ** 2)
        variances = np.var(magnitude, axis=(1, 2))
    else:
        variances = np.var(volume, axis=(1, 2, 3))

    num_to_remove = int(num_slices * removal_percentage)
    if num_to_remove >= num_slices:
        num_to_remove = max(0, num_slices - 1)

    sorted_indices = np.argsort(variances)
    remove_indices = sorted_indices[:num_to_remove]
    keep_indices = np.sort(sorted_indices[num_to_remove:])

    return volume[keep_indices], remove_indices, keep_indices


def down_sample_i(np_i_data: np.ndarray, scale: int = 2) -> np.ndarray:
    if np_i_data.ndim != 4 or np_i_data.shape[-1] != 2:
        raise ValueError("np_i_data must have shape [N,H,W,2]")
    if scale <= 0:
        raise ValueError("scale must be positive")

    tensor = torch.from_numpy(np_i_data).to(torch.float32)
    tensor = tensor.permute(0, 3, 1, 2)
    downsampled = F.avg_pool2d(tensor, kernel_size=scale, stride=scale)
    return downsampled.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
