from __future__ import annotations

import numpy as np
import torch


def build_normalized_grid(resolution: tuple[int, int]) -> torch.Tensor:
    height, width = resolution
    if height <= 0 or width <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")

    ranges = [np.linspace(0.0, 1.0, num=height), np.linspace(0.0, 1.0, num=width)]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    stacked = np.stack(grid, axis=-1).astype(np.float32)
    return torch.from_numpy(stacked)


def enumerate_positions(resolution: tuple[int, int]) -> torch.Tensor:
    height, width = resolution
    rows = torch.arange(height, dtype=torch.int64)
    cols = torch.arange(width, dtype=torch.int64)
    pos = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1)
    return pos.reshape(-1, 2)


def build_lr_positions(resolution: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    grid = build_normalized_grid(resolution)
    pos = enumerate_positions(resolution)
    pos_norm = grid[pos[:, 0], pos[:, 1]]
    return pos, pos_norm
