from __future__ import annotations

import pytest

from training.logger import TensorboardLogger, build_metric_key


class _DummyWriter:
    def __init__(self) -> None:
        self.items: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self.items.append((tag, scalar_value, global_step))


def test_metric_keys_are_unique_across_metric_names() -> None:
    psnr_key = build_metric_key(split="valid", stage="K", family="metrics", metric="psnr")
    ssim_key = build_metric_key(split="valid", stage="K", family="metrics", metric="ssim")
    assert psnr_key != ssim_key


def test_tensorboard_logger_uses_distinct_metric_tags() -> None:
    writer = _DummyWriter()
    logger = TensorboardLogger(writer)

    key1 = logger.log_metric(
        split="valid",
        stage="K",
        family="metrics",
        metric="psnr",
        value=32.1,
        step=1,
    )
    key2 = logger.log_metric(
        split="valid",
        stage="K",
        family="metrics",
        metric="ssim",
        value=0.91,
        step=1,
    )

    assert key1 != key2
    assert writer.items[0][0] == key1
    assert writer.items[1][0] == key2


def test_metric_key_requires_non_empty_components() -> None:
    with pytest.raises(ValueError):
        build_metric_key(split="", stage="K", family="metrics", metric="psnr")

