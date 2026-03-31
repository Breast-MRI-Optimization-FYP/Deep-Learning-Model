from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch


def resolve_device(gpu: str | None = None, *, prefer_cuda: bool = True) -> torch.device:
    if gpu is not None and str(gpu).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)

    if isinstance(value, Mapping):
        return {key: to_device(item, device) for key, item in value.items()}

    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)

    if isinstance(value, list):
        return [to_device(item, device) for item in value]

    return value
