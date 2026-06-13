from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.grids import build_lr_positions, build_normalized_grid
from data.masks import ensure_mask_channels
from data.sequence import SequenceLimiter
from data.tokenize import tokenize_kspace_slice
from utils.fftc import ifft2c

from .schemas import MaskMetadata


class EvaluationKSpaceDataset(Dataset):
    """Deterministic cross-product of test slices and explicitly selected masks."""

    def __init__(
        self,
        *,
        hr_data_path: str | Path,
        mask_path: str | Path,
        mask_metadata: list[MaskMetadata],
        lr_resolution: tuple[int, int],
        max_seq_len: int | None = None,
        max_samples: int | None = None,
    ) -> None:
        hr_np = np.load(str(hr_data_path)).astype(np.float32)
        if hr_np.ndim != 4 or hr_np.shape[-1] != 2:
            raise ValueError("hr_data must have shape [N,H,W,2]")
        if max_samples is not None:
            hr_np = hr_np[:max_samples]

        self.k_gt = torch.from_numpy(hr_np)
        self.i_gt = ifft2c(self.k_gt, need_shift=True)
        self.hr_grid = build_normalized_grid((hr_np.shape[1], hr_np.shape[2]))
        self.lr_pos, self.lr_pos_norm = build_lr_positions(lr_resolution)

        self.mask_bank = ensure_mask_channels(np.load(str(mask_path)))
        if self.mask_bank.shape[1:3] != hr_np.shape[1:3]:
            raise ValueError("mask spatial shape must match hr_data spatial shape")

        for metadata in mask_metadata:
            if metadata.mask_id < 0 or metadata.mask_id >= self.mask_bank.shape[0]:
                raise ValueError(f"Mask id out of range: {metadata.mask_id}")
        self.mask_metadata = list(mask_metadata)
        self.sequence_limiter = SequenceLimiter(max_seq_len) if max_seq_len is not None else None

    def __len__(self) -> int:
        return int(self.k_gt.shape[0]) * len(self.mask_metadata)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample_index = index // len(self.mask_metadata)
        metadata = self.mask_metadata[index % len(self.mask_metadata)]
        mask = self.mask_bank[metadata.mask_id]

        sample = tokenize_kspace_slice(self.k_gt[sample_index], mask, self.hr_grid)
        sample.i_gt = self.i_gt[sample_index].to(torch.float32)
        sample.k_gt = self.k_gt[sample_index].to(torch.float32)
        sample.lr_pos = self.lr_pos
        sample.lr_pos_norm = self.lr_pos_norm
        if self.sequence_limiter is not None:
            sample = self.sequence_limiter.truncate(sample)

        return {
            "sampled_k": sample.sampled_k.to(torch.float32),
            "sampled_pos": sample.sampled_pos.to(torch.int64),
            "sampled_pos_norm": sample.sampled_pos_norm.to(torch.float32),
            "unsampled_pos_norm": sample.unsampled_pos_norm.to(torch.float32),
            "unsampled_pos": sample.unsampled_pos.to(torch.int64),
            "k_us": sample.k_us.to(torch.float32),
            "i_gt": sample.i_gt.to(torch.float32),
            "k_gt": sample.k_gt.to(torch.float32),
            "selected_mask": sample.selected_mask.to(torch.float32),
            "LR_pos": self.lr_pos.to(torch.int64),
            "LR_pos_norm": self.lr_pos_norm.to(torch.float32),
            "sample_index": torch.tensor(sample_index, dtype=torch.int64),
            "mask_id": torch.tensor(metadata.mask_id, dtype=torch.int64),
            "acceleration": torch.tensor(metadata.acceleration, dtype=torch.float32),
            "actual_acceleration": torch.tensor(
                metadata.actual_acceleration or metadata.acceleration,
                dtype=torch.float32,
            ),
        }
