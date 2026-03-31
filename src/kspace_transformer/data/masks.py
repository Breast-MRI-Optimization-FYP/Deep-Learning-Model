from __future__ import annotations

import numpy as np
import torch


def ensure_mask_channels(mask_bank: np.ndarray | torch.Tensor) -> np.ndarray:
    mask_np = np.asarray(mask_bank)
    if mask_np.ndim == 3:
        mask_np = np.repeat(mask_np[:, :, :, np.newaxis], 2, axis=-1)
    elif mask_np.ndim == 4 and mask_np.shape[-1] == 1:
        mask_np = np.repeat(mask_np, 2, axis=-1)
    elif mask_np.ndim != 4 or mask_np.shape[-1] != 2:
        raise ValueError(
            "mask_bank must have shape [N,H,W], [N,H,W,1], or [N,H,W,2]"
        )
    return mask_np.astype(bool)


def select_mask(mask_bank: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    if mask_bank.ndim != 4 or mask_bank.shape[-1] != 2:
        raise ValueError("mask_bank must have shape [N,H,W,2]")
    if mask_bank.shape[0] == 0:
        raise ValueError("mask_bank cannot be empty")

    picker = rng if rng is not None else np.random.default_rng()
    index = int(picker.integers(low=0, high=mask_bank.shape[0]))
    return mask_bank[index].astype(bool)


def apply_undersampling(
    k_gt_slice: torch.Tensor,
    mask_sampled: np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask_t = torch.as_tensor(mask_sampled, dtype=torch.bool, device=k_gt_slice.device)
    if mask_t.ndim == 2:
        mask_t = mask_t.unsqueeze(-1).expand(-1, -1, 2)
    if mask_t.ndim != 3 or mask_t.shape[-1] != 2:
        raise ValueError("mask_sampled must have shape [H,W] or [H,W,2]")

    k_us = k_gt_slice.clone()
    unsampled_mask = ~mask_t
    k_us[unsampled_mask] = 0
    return k_us, unsampled_mask


def extract_positions(
    mask_sampled: np.ndarray | torch.Tensor,
    hr_grid: torch.Tensor,
) -> tuple[np.ndarray, torch.Tensor, np.ndarray, torch.Tensor]:
    mask_np = np.asarray(mask_sampled).astype(bool)
    if mask_np.ndim == 2:
        mask_np = np.repeat(mask_np[:, :, np.newaxis], 2, axis=-1)
    if mask_np.ndim != 3 or mask_np.shape[-1] != 2:
        raise ValueError("mask_sampled must have shape [H,W] or [H,W,2]")

    sampled_idx = np.nonzero(mask_np[:, :, 0])
    unsampled_idx = np.nonzero(~mask_np[:, :, 0])

    sampled_pos = np.stack(sampled_idx, axis=-1).astype(np.int64)
    unsampled_pos = np.stack(unsampled_idx, axis=-1).astype(np.int64)

    sampled_pos_norm = hr_grid[sampled_idx]
    unsampled_pos_norm = hr_grid[unsampled_idx]
    return sampled_pos, sampled_pos_norm, unsampled_pos, unsampled_pos_norm
