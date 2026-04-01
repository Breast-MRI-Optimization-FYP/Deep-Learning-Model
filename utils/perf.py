from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class RuntimeTracker:
    device: torch.device | None = None
    start_time: float = 0.0

    def start(self) -> None:
        self.start_time = time.perf_counter()
        if self.device is not None and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def elapsed_seconds(self) -> float:
        if self.start_time == 0.0:
            return 0.0
        return float(time.perf_counter() - self.start_time)

    def peak_memory_bytes(self) -> int | None:
        if self.device is None:
            return None
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated(self.device))
