from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class MaskMetadata:
    mask_id: int
    acceleration: float
    sampling_pattern: str = "unknown"
    actual_acceleration: float | None = None

    @property
    def acceleration_label(self) -> str:
        rounded = round(self.acceleration)
        if abs(self.acceleration - rounded) < 1e-6:
            return f"x{rounded}"
        return f"x{self.acceleration:g}"


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    output_dir: Path
    stages: tuple[str, ...] = ("K", "RM")
    up_scale: int = 2
    conv_weight: float = 1.0
    qualitative_samples_per_acceleration: int = 1

    def validate(self) -> None:
        if not self.stages:
            raise ValueError("At least one evaluation stage is required")
        if any(stage not in {"K", "RM"} for stage in self.stages):
            raise ValueError("Evaluation stages must be K and/or RM")
        if self.up_scale <= 0:
            raise ValueError("up_scale must be positive")
        if self.qualitative_samples_per_acceleration < 0:
            raise ValueError("qualitative_samples_per_acceleration must be non-negative")


@dataclass(frozen=True, slots=True)
class SampleMetricRecord:
    sample_index: int
    mask_id: int
    acceleration: float
    acceleration_label: str
    actual_acceleration: float
    sampling_pattern: str
    stage: str
    psnr: float
    ssim: float
    nmse: float


@dataclass(slots=True)
class QualitativeSample:
    sample_index: int
    mask_id: int
    acceleration_label: str
    undersampled: np.ndarray
    ground_truth: np.ndarray
    predictions: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationResult:
    records: list[SampleMetricRecord]
    qualitative_samples: list[QualitativeSample]
    runtime_seconds: float
    peak_memory_bytes: int | None
