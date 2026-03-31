from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from skimage.metrics import structural_similarity as structural_similarity


def _to_numpy(data: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    raise TypeError(f"Expected numpy array or torch tensor, got {type(data)!r}")


def _to_magnitude(images: np.ndarray) -> np.ndarray:
    if images.ndim != 4 or images.shape[-1] != 2:
        raise ValueError("Expected shape [B,H,W,2] for complex-valued images")
    return np.abs(images[:, :, :, 0] + 1j * images[:, :, :, 1])


def compute_psnr(
    pred_i: np.ndarray | torch.Tensor,
    i_gt: np.ndarray | torch.Tensor,
    *,
    eps: float = 1e-12,
) -> float:
    pred_np = _to_numpy(pred_i).astype(np.float64, copy=False)
    gt_np = _to_numpy(i_gt).astype(np.float64, copy=False)
    if pred_np.shape != gt_np.shape:
        raise ValueError("pred_i and i_gt must have the same shape")

    pred_mag = _to_magnitude(pred_np)
    gt_mag = _to_magnitude(gt_np)

    total_psnr = 0.0
    for idx in range(gt_mag.shape[0]):
        data_range = float(gt_mag[idx].max() - gt_mag[idx].min())
        mse = float(np.mean((pred_mag[idx] - gt_mag[idx]) ** 2))

        if mse <= eps:
            total_psnr += float("inf")
            continue
        if data_range <= eps:
            total_psnr += 0.0
            continue

        total_psnr += float(10.0 * np.log10((data_range ** 2) / mse))

    return total_psnr


def compute_ssim(
    pred_i: np.ndarray | torch.Tensor,
    i_gt: np.ndarray | torch.Tensor,
    *,
    eps: float = 1e-12,
) -> float:
    pred_np = _to_numpy(pred_i).astype(np.float64, copy=False)
    gt_np = _to_numpy(i_gt).astype(np.float64, copy=False)
    if pred_np.shape != gt_np.shape:
        raise ValueError("pred_i and i_gt must have the same shape")

    pred_mag = _to_magnitude(pred_np)
    gt_mag = _to_magnitude(gt_np)

    total_ssim = 0.0
    for idx in range(gt_mag.shape[0]):
        data_range = float(gt_mag[idx].max() - gt_mag[idx].min())

        if data_range <= eps:
            total_ssim += 1.0 if np.allclose(pred_mag[idx], gt_mag[idx]) else 0.0
            continue

        total_ssim += float(
            structural_similarity(gt_mag[idx], pred_mag[idx], data_range=data_range)
        )

    return total_ssim


@dataclass(slots=True)
class MetricsAccumulator:
    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def update(self, name: str, value: float, *, n: int = 1) -> None:
        if n <= 0:
            raise ValueError("n must be positive")
        self.totals[name] = self.totals.get(name, 0.0) + float(value)
        self.counts[name] = self.counts.get(name, 0) + int(n)

    def mean(self, name: str) -> float:
        if name not in self.totals or name not in self.counts:
            raise KeyError(f"Metric {name!r} has not been tracked")
        return self.totals[name] / self.counts[name]

    def as_dict(self) -> dict[str, float]:
        return {name: self.mean(name) for name in self.totals}
