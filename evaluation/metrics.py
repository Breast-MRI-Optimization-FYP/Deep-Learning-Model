from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity

ArrayLike = np.ndarray | torch.Tensor


def to_numpy(data: ArrayLike) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    raise TypeError(f"Expected numpy array or torch tensor, got {type(data)!r}")


def magnitude_images(images: ArrayLike) -> np.ndarray:
    array = to_numpy(images).astype(np.float64, copy=False)
    if array.ndim != 4 or array.shape[-1] != 2:
        raise ValueError("Expected complex-valued images with shape [B,H,W,2]")
    return np.hypot(array[..., 0], array[..., 1])


def _paired_magnitudes(pred_i: ArrayLike, i_gt: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    pred_mag = magnitude_images(pred_i)
    gt_mag = magnitude_images(i_gt)
    if pred_mag.shape != gt_mag.shape:
        raise ValueError("pred_i and i_gt must have the same shape")
    return pred_mag, gt_mag


def per_sample_psnr(
    pred_i: ArrayLike,
    i_gt: ArrayLike,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    pred_mag, gt_mag = _paired_magnitudes(pred_i, i_gt)
    values = np.empty(gt_mag.shape[0], dtype=np.float64)

    for idx in range(gt_mag.shape[0]):
        data_range = float(np.ptp(gt_mag[idx]))
        mse = float(np.mean((pred_mag[idx] - gt_mag[idx]) ** 2))
        if mse <= eps:
            values[idx] = float("inf")
        elif data_range <= eps:
            values[idx] = 0.0
        else:
            values[idx] = 10.0 * np.log10((data_range**2) / mse)
    return values


def per_sample_ssim(
    pred_i: ArrayLike,
    i_gt: ArrayLike,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    pred_mag, gt_mag = _paired_magnitudes(pred_i, i_gt)
    values = np.empty(gt_mag.shape[0], dtype=np.float64)

    for idx in range(gt_mag.shape[0]):
        data_range = float(np.ptp(gt_mag[idx]))
        if data_range <= eps:
            values[idx] = 1.0 if np.allclose(pred_mag[idx], gt_mag[idx]) else 0.0
        else:
            values[idx] = structural_similarity(
                gt_mag[idx],
                pred_mag[idx],
                data_range=data_range,
            )
    return values


def per_sample_nmse(
    pred_i: ArrayLike,
    i_gt: ArrayLike,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return magnitude-image NMSE for each sample; lower is better."""
    pred_mag, gt_mag = _paired_magnitudes(pred_i, i_gt)
    error_energy = np.sum((pred_mag - gt_mag) ** 2, axis=(1, 2))
    target_energy = np.sum(gt_mag**2, axis=(1, 2))

    values = np.empty(gt_mag.shape[0], dtype=np.float64)
    nonzero = target_energy > eps
    values[nonzero] = error_energy[nonzero] / target_energy[nonzero]
    values[~nonzero] = np.where(error_energy[~nonzero] <= eps, 0.0, float("inf"))
    return values


def compute_psnr(pred_i: ArrayLike, i_gt: ArrayLike, *, eps: float = 1e-12) -> float:
    """Compatibility API returning the sum of per-sample PSNR values."""
    return float(np.sum(per_sample_psnr(pred_i, i_gt, eps=eps)))


def compute_ssim(pred_i: ArrayLike, i_gt: ArrayLike, *, eps: float = 1e-12) -> float:
    """Compatibility API returning the sum of per-sample SSIM values."""
    return float(np.sum(per_sample_ssim(pred_i, i_gt, eps=eps)))


def compute_nmse(pred_i: ArrayLike, i_gt: ArrayLike, *, eps: float = 1e-12) -> float:
    """Compatibility API returning the sum of per-sample NMSE values."""
    return float(np.sum(per_sample_nmse(pred_i, i_gt, eps=eps)))
