from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class CheckpointManager:
    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric: float | None = None
        self.best_path: Path | None = None

    def save_checkpoint(self, state: dict[str, Any], filename: str = "checkpoint.pth") -> Path:
        path = self.checkpoint_dir / filename
        torch.save(state, path)
        return path

    def save_best(
        self,
        state: dict[str, Any],
        *,
        metric: float,
        mode: str = "max",
        filename: str = "best.pth",
    ) -> tuple[bool, Path]:
        if mode not in {"max", "min"}:
            raise ValueError("mode must be either 'max' or 'min'")

        improved = False
        if self.best_metric is None:
            improved = True
        elif mode == "max":
            improved = metric > self.best_metric
        else:
            improved = metric < self.best_metric

        path = self.checkpoint_dir / filename
        if improved:
            torch.save(state, path)
            self.best_metric = float(metric)
            self.best_path = path

        return improved, path

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> dict[str, Any]:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location=map_location)
        if not isinstance(checkpoint, dict):
            raise TypeError("Checkpoint payload must be a dictionary")
        return checkpoint

    def validate_state_dict(
        self,
        model: nn.Module,
        state_dict: dict[str, torch.Tensor],
        *,
        strict: bool = False,
    ) -> tuple[list[str], list[str]]:
        model_keys = set(model.state_dict().keys())
        incoming_keys = set(state_dict.keys())

        missing_keys = sorted(model_keys - incoming_keys)
        unexpected_keys = sorted(incoming_keys - model_keys)

        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                "State dict mismatch. "
                f"Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}."
            )

        return missing_keys, unexpected_keys

    def load_model_state(
        self,
        model: nn.Module,
        checkpoint: dict[str, Any],
        *,
        state_dict_key: str = "model_state_dict",
        strict: bool = True,
    ) -> tuple[list[str], list[str]]:
        if state_dict_key not in checkpoint:
            raise KeyError(f"Checkpoint missing key: {state_dict_key}")

        state_dict = checkpoint[state_dict_key]
        if not isinstance(state_dict, dict):
            raise TypeError("model_state_dict must be a dictionary")

        if strict:
            model.load_state_dict(state_dict, strict=True)
            return [], []

        incompatible = model.load_state_dict(state_dict, strict=False)
        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
