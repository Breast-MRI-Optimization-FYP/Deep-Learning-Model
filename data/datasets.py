from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils.fftc import ifft2c

from .grids import build_lr_positions, build_normalized_grid
from .masks import ensure_mask_channels, select_mask
from .sequence import SequenceLimiter, TruncationStats
from .tokenize import tokenize_kspace_slice
from .types import DatasetCache


class BaseKSpaceDataset(Dataset):
    def __init__(
        self,
        *,
        hr_data_path: str | Path,
        mask_path: str | Path,
        lr_data_path: str | Path | None = None,
        lr_resolution: tuple[int, int] | None = None,
        seed: int = 42,
        max_seq_len: int | None = None,
        max_samples: int | None = None,
        precompute: bool = True,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.sequence_limiter = SequenceLimiter(max_seq_len) if max_seq_len is not None else None
        self.truncation_stats = TruncationStats()

        hr_np = np.load(str(hr_data_path)).astype(np.float32)
        if hr_np.ndim != 4 or hr_np.shape[-1] != 2:
            raise ValueError("hr_data must have shape [N,H,W,2]")
        if max_samples is not None:
            hr_np = hr_np[:max_samples]

        self.k_gt = torch.from_numpy(hr_np)
        self.i_gt = ifft2c(self.k_gt, need_shift=True)
        self.hr_grid = build_normalized_grid((self.k_gt.shape[1], self.k_gt.shape[2]))

        mask_np = np.load(str(mask_path))
        self.mask_bank = ensure_mask_channels(mask_np)
        if self.mask_bank.shape[1:3] != self.k_gt.shape[1:3]:
            raise ValueError("mask spatial shape must match hr_data spatial shape")

        sampled_num = int(self.mask_bank[0, :, :, 0].sum())
        self.sampled_num = sampled_num
        self.unsampled_num = int(self.k_gt.shape[1] * self.k_gt.shape[2] - sampled_num)

        self.lr_k_gt: torch.Tensor | None = None
        self.lr_i_gt: torch.Tensor | None = None
        if lr_data_path is not None:
            lr_np = np.load(str(lr_data_path)).astype(np.float32)
            if max_samples is not None:
                lr_np = lr_np[:max_samples]
            if lr_np.shape[0] != self.k_gt.shape[0]:
                raise ValueError("lr_data sample count must match hr_data sample count")
            self.lr_k_gt = torch.from_numpy(lr_np)
            self.lr_i_gt = ifft2c(self.lr_k_gt, need_shift=True)
            lr_resolution_final = (self.lr_k_gt.shape[1], self.lr_k_gt.shape[2])
        else:
            if lr_resolution is None:
                raise ValueError("lr_resolution must be provided when lr_data_path is None")
            lr_resolution_final = lr_resolution

        self.lr_pos, self.lr_pos_norm = build_lr_positions(lr_resolution_final)

        self.cache = DatasetCache()
        self.k_us_cache: list[torch.Tensor] = []
        self.selected_mask_cache: list[torch.Tensor] = []

        if precompute:
            self.reassign_mask()

    def __len__(self) -> int:
        if self.k_us_cache:
            return len(self.k_us_cache)
        return int(self.k_gt.shape[0])

    def _build_sample(self, index: int):
        mask = select_mask(self.mask_bank, self.rng)
        sample = tokenize_kspace_slice(self.k_gt[index], mask, self.hr_grid)
        sample.i_gt = self.i_gt[index].to(torch.float32)
        sample.k_gt = self.k_gt[index].to(torch.float32)
        sample.lr_pos = self.lr_pos
        sample.lr_pos_norm = self.lr_pos_norm

        if self.lr_i_gt is not None and self.lr_k_gt is not None:
            sample.lr_i_gt = self.lr_i_gt[index].to(torch.float32)
            sample.lr_k_gt = self.lr_k_gt[index].to(torch.float32)

        if self.sequence_limiter is not None:
            sample = self.sequence_limiter.truncate(sample, stats=self.truncation_stats)
        else:
            self.truncation_stats.total_samples += 1
            self.truncation_stats.max_observed_len = max(
                self.truncation_stats.max_observed_len,
                int(sample.sampled_k.shape[0]),
                int(sample.unsampled_pos.shape[0]),
            )
            self.truncation_stats.sampled_tokens_before += int(sample.sampled_k.shape[0])
            self.truncation_stats.sampled_tokens_after += int(sample.sampled_k.shape[0])

        sample.validate()
        return sample

    def reassign_mask(self) -> None:
        self.cache = DatasetCache()
        self.k_us_cache = []
        self.selected_mask_cache = []

        for index in range(int(self.k_gt.shape[0])):
            sample = self._build_sample(index)
            self.cache.sampled_k.append(sample.sampled_k.cpu())
            self.cache.sampled_pos.append(sample.sampled_pos.cpu())
            self.cache.sampled_pos_norm.append(sample.sampled_pos_norm.cpu())
            self.cache.unsampled_pos.append(sample.unsampled_pos.cpu())
            self.cache.unsampled_pos_norm.append(sample.unsampled_pos_norm.cpu())
            self.k_us_cache.append(sample.k_us.cpu())
            self.selected_mask_cache.append(sample.selected_mask.cpu())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not self.k_us_cache:
            self.reassign_mask()

        result: dict[str, torch.Tensor] = {
            "sampled_k": self.cache.sampled_k[index].to(torch.float32),
            "sampled_pos": self.cache.sampled_pos[index].to(torch.int64),
            "sampled_pos_norm": self.cache.sampled_pos_norm[index].to(torch.float32),
            "unsampled_pos_norm": self.cache.unsampled_pos_norm[index].to(torch.float32),
            "unsampled_pos": self.cache.unsampled_pos[index].to(torch.int64),
            "k_us": self.k_us_cache[index].to(torch.float32),
            "i_gt": self.i_gt[index].to(torch.float32),
            "k_gt": self.k_gt[index].to(torch.float32),
            "selected_mask": self.selected_mask_cache[index].to(torch.float32),
            "LR_pos": self.lr_pos.to(torch.int64),
            "LR_pos_norm": self.lr_pos_norm.to(torch.float32),
        }

        if self.lr_i_gt is not None and self.lr_k_gt is not None:
            result["LR_i_gt"] = self.lr_i_gt[index].to(torch.float32)
            result["LR_k_gt"] = self.lr_k_gt[index].to(torch.float32)

        return result


class TrainKSpaceDataset(BaseKSpaceDataset):
    def __init__(
        self,
        hr_data_path: str | Path,
        lr_data_path: str | Path,
        mask_path: str | Path,
        *,
        seed: int = 42,
        max_seq_len: int | None = None,
        max_samples: int | None = None,
        precompute: bool = True,
    ) -> None:
        super().__init__(
            hr_data_path=hr_data_path,
            lr_data_path=lr_data_path,
            mask_path=mask_path,
            seed=seed,
            max_seq_len=max_seq_len,
            max_samples=max_samples,
            precompute=precompute,
        )


class ValidKSpaceDataset(BaseKSpaceDataset):
    def __init__(
        self,
        hr_data_path: str | Path,
        lr_data_path: str | Path,
        mask_path: str | Path,
        *,
        seed: int = 42,
        max_seq_len: int | None = None,
        max_samples: int | None = None,
        precompute: bool = True,
    ) -> None:
        super().__init__(
            hr_data_path=hr_data_path,
            lr_data_path=lr_data_path,
            mask_path=mask_path,
            seed=seed,
            max_seq_len=max_seq_len,
            max_samples=max_samples,
            precompute=precompute,
        )


class TestKSpaceDataset(BaseKSpaceDataset):
    __test__ = False

    def __init__(
        self,
        hr_data_path: str | Path,
        mask_path: str | Path,
        *,
        lr_resolution: tuple[int, int] = (64, 64),
        seed: int = 42,
        max_seq_len: int | None = None,
        max_samples: int | None = None,
        precompute: bool = True,
    ) -> None:
        super().__init__(
            hr_data_path=hr_data_path,
            lr_data_path=None,
            lr_resolution=lr_resolution,
            mask_path=mask_path,
            seed=seed,
            max_seq_len=max_seq_len,
            max_samples=max_samples,
            precompute=precompute,
        )


class KSpaceCollator:
    def __init__(self, max_seq_len: int = 8000) -> None:
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.max_seq_len = max_seq_len

    def _truncate(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.shape[0] <= self.max_seq_len:
            return seq
        return seq[: self.max_seq_len]

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("batch cannot be empty")

        sampled_k_list = [self._truncate(item["sampled_k"]) for item in batch]
        sampled_pos_list = [self._truncate(item["sampled_pos"]) for item in batch]
        sampled_pos_norm_list = [self._truncate(item["sampled_pos_norm"]) for item in batch]
        unsampled_pos_norm_list = [self._truncate(item["unsampled_pos_norm"]) for item in batch]
        unsampled_pos_list = [self._truncate(item["unsampled_pos"]) for item in batch]

        sampled_k_padded = pad_sequence(sampled_k_list, batch_first=True)
        sampled_pos_padded = pad_sequence(sampled_pos_list, batch_first=True)
        sampled_pos_norm_padded = pad_sequence(sampled_pos_norm_list, batch_first=True)
        unsampled_pos_norm_padded = pad_sequence(unsampled_pos_norm_list, batch_first=True)
        unsampled_pos_padded = pad_sequence(unsampled_pos_list, batch_first=True)

        batch_size = len(batch)
        lr_pos = batch[0]["LR_pos"]
        lr_pos_norm = batch[0]["LR_pos_norm"]

        result: dict[str, torch.Tensor] = {
            "sampled_k": sampled_k_padded,
            "sampled_pos": sampled_pos_padded,
            "sampled_pos_norm": sampled_pos_norm_padded,
            "unsampled_pos_norm": unsampled_pos_norm_padded,
            "unsampled_pos": unsampled_pos_padded,
            "k_us": torch.stack([item["k_us"] for item in batch]),
            "i_gt": torch.stack([item["i_gt"] for item in batch]),
            "k_gt": torch.stack([item["k_gt"] for item in batch]),
            "selected_mask": torch.stack([item["selected_mask"] for item in batch]),
            "LR_pos": lr_pos.unsqueeze(0).expand(batch_size, -1, -1).clone(),
            "LR_pos_norm": lr_pos_norm.unsqueeze(0).expand(batch_size, -1, -1).clone(),
        }

        if all("LR_i_gt" in item for item in batch):
            result["LR_i_gt"] = torch.stack([item["LR_i_gt"] for item in batch])
        if all("LR_k_gt" in item for item in batch):
            result["LR_k_gt"] = torch.stack([item["LR_k_gt"] for item in batch])

        return result

