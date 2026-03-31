from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ScalarWriter(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        ...


def build_metric_key(*, split: str, stage: str, family: str, metric: str) -> str:
    parts = [split.strip().lower(), stage.strip().upper(), family.strip().lower(), metric.strip().lower()]
    if any(not part for part in parts):
        raise ValueError("split, stage, family, and metric must all be non-empty")
    return "/".join(parts)


@dataclass(slots=True)
class RunLogger:
    prefix: str = ""

    def info(self, message: str) -> None:
        if self.prefix:
            print(f"[{self.prefix}] {message}")
            return
        print(message)


class TensorboardLogger:
    def __init__(self, writer: ScalarWriter) -> None:
        self.writer = writer

    def log_metric(
        self,
        *,
        split: str,
        stage: str,
        family: str,
        metric: str,
        value: float,
        step: int,
    ) -> str:
        key = build_metric_key(split=split, stage=stage, family=family, metric=metric)
        self.writer.add_scalar(key, float(value), int(step))
        return key
