from __future__ import annotations

from dataclasses import dataclass, field

from evaluation.metrics import (
    compute_nmse,
    compute_psnr,
    compute_ssim,
    magnitude_images,
)

_to_magnitude = magnitude_images


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
