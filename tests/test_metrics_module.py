from __future__ import annotations

import numpy as np

from training.metrics import (
    MetricsAccumulator,
    compute_psnr,
    compute_ssim,
)


def test_psnr_is_inf_for_identical_batches() -> None:
    gt = np.zeros((2, 8, 8, 2), dtype=np.float32)
    pred = gt.copy()

    psnr_sum = compute_psnr(pred, gt)
    assert np.isinf(psnr_sum)


def test_psnr_is_finite_for_non_identical_batches() -> None:
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(2, 8, 8, 2)).astype(np.float32)
    pred = gt + 0.05

    psnr_sum = compute_psnr(pred, gt)
    assert np.isfinite(psnr_sum)


def test_ssim_handles_zero_data_range() -> None:
    gt = np.zeros((2, 8, 8, 2), dtype=np.float32)
    pred_same = np.zeros((2, 8, 8, 2), dtype=np.float32)
    pred_diff = np.ones((2, 8, 8, 2), dtype=np.float32)

    assert compute_ssim(pred_same, gt) == 2.0
    assert compute_ssim(pred_diff, gt) == 0.0


def test_metrics_accumulator_mean_tracking() -> None:
    acc = MetricsAccumulator()
    acc.update("psnr", 20.0, n=2)
    acc.update("psnr", 10.0, n=1)
    acc.update("ssim", 0.9, n=3)

    assert acc.mean("psnr") == 10.0
    assert acc.mean("ssim") == 0.3

    as_dict = acc.as_dict()
    assert as_dict["psnr"] == 10.0
    assert as_dict["ssim"] == 0.3

