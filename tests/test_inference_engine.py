from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from data.datasets import KSpaceCollator
from inference.engine import InferenceRunner
from model import KSpaceTransformer
from training.stage import TrainingStage


class _MiniDataset(Dataset):
    def __init__(self, size: int = 4) -> None:
        self.size = size
        self.lr_pos = torch.stack(
            torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij"), dim=-1
        ).reshape(-1, 2)
        self.lr_pos_norm = torch.rand(16, 2)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sampled_len = 12 + (index % 2)
        unsampled_len = 10 + (index % 3)
        return {
            "sampled_k": torch.randn(sampled_len, 2),
            "sampled_pos": torch.randint(0, 8, (sampled_len, 2), dtype=torch.int64),
            "sampled_pos_norm": torch.randn(sampled_len, 2),
            "unsampled_pos_norm": torch.randn(unsampled_len, 2),
            "unsampled_pos": torch.randint(0, 8, (unsampled_len, 2), dtype=torch.int64),
            "k_us": torch.randn(8, 8, 2),
            "i_gt": torch.randn(8, 8, 2),
            "k_gt": torch.randn(8, 8, 2),
            "selected_mask": torch.zeros(8, 8, 2),
            "LR_i_gt": torch.randn(4, 4, 2),
            "LR_k_gt": torch.randn(4, 4, 2),
            "LR_pos": self.lr_pos,
            "LR_pos_norm": self.lr_pos_norm,
        }


def _build_loader() -> DataLoader[Any]:
    ds = _MiniDataset(size=4)
    collator = KSpaceCollator(max_seq_len=32)
    return DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collator)


def _build_model() -> KSpaceTransformer:
    return KSpaceTransformer(
        lr_size=4,
        channel=2,
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        num_lrdecoder_layers=2,
        num_hrdecoder_layers=2,
        dim_feedforward=64,
        hr_conv_channel=16,
        hr_conv_num=1,
        hr_kernel_size=3,
        dropout=0.0,
        activation="relu",
    )


def test_inference_runner_for_k_and_rm_stages() -> None:
    loader = _build_loader()
    model = _build_model()
    runner = InferenceRunner(model=model, device=torch.device("cpu"))

    summary_k = runner.run_test_set(loader, stage=TrainingStage.K)
    summary_rm = runner.run_test_set(loader, stage=TrainingStage.RM)

    assert summary_k["stage"] == "K"
    assert summary_rm["stage"] == "RM"
    assert summary_k["num_samples"] == 4
    assert summary_rm["num_samples"] == 4
    assert torch.isfinite(torch.tensor(float(summary_k["mean_ssim"])))
    assert torch.isfinite(torch.tensor(float(summary_rm["mean_ssim"])))
    assert torch.isfinite(torch.tensor(float(summary_k["mean_nmse"])))
    assert torch.isfinite(torch.tensor(float(summary_rm["mean_nmse"])))

