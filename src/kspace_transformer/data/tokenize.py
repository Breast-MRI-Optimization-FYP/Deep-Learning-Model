from __future__ import annotations

import numpy as np
import torch

from .masks import apply_undersampling, extract_positions
from .types import TokenizedSample


def _gather_sampled_k(k_gt_slice: torch.Tensor, sampled_pos: np.ndarray) -> torch.Tensor:
    if sampled_pos.shape[0] == 0:
        return torch.empty((0, 2), dtype=k_gt_slice.dtype, device=k_gt_slice.device)

    rows = torch.from_numpy(sampled_pos[:, 0]).to(device=k_gt_slice.device, dtype=torch.int64)
    cols = torch.from_numpy(sampled_pos[:, 1]).to(device=k_gt_slice.device, dtype=torch.int64)
    real = k_gt_slice[rows, cols, 0]
    imag = k_gt_slice[rows, cols, 1]
    return torch.stack((real, imag), dim=-1)


def tokenize_kspace_slice(
    k_gt_slice: torch.Tensor,
    mask_sampled: np.ndarray | torch.Tensor,
    hr_grid: torch.Tensor,
) -> TokenizedSample:
    if k_gt_slice.ndim != 3 or k_gt_slice.shape[-1] != 2:
        raise ValueError("k_gt_slice must have shape [H,W,2]")

    k_us, selected_mask = apply_undersampling(k_gt_slice, mask_sampled)
    sampled_pos, sampled_pos_norm, unsampled_pos, unsampled_pos_norm = extract_positions(
        mask_sampled, hr_grid
    )

    sample = TokenizedSample(
        sampled_k=_gather_sampled_k(k_gt_slice, sampled_pos),
        sampled_pos=torch.from_numpy(sampled_pos).to(torch.int64),
        sampled_pos_norm=sampled_pos_norm.to(torch.float32),
        unsampled_pos=torch.from_numpy(unsampled_pos).to(torch.int64),
        unsampled_pos_norm=unsampled_pos_norm.to(torch.float32),
        k_us=k_us.to(torch.float32),
        selected_mask=selected_mask,
        k_gt=k_gt_slice.to(torch.float32),
    )
    sample.validate()
    return sample
