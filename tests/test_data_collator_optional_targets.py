from __future__ import annotations

import torch

from data.datasets import KSpaceCollator


def test_collator_handles_batches_without_lr_targets() -> None:
    lr_pos = torch.stack(
        torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij"), dim=-1
    ).reshape(-1, 2)
    lr_pos_norm = torch.rand(16, 2)

    sample = {
        "sampled_k": torch.randn(12, 2),
        "sampled_pos": torch.randint(0, 8, (12, 2), dtype=torch.int64),
        "sampled_pos_norm": torch.randn(12, 2),
        "unsampled_pos_norm": torch.randn(10, 2),
        "unsampled_pos": torch.randint(0, 8, (10, 2), dtype=torch.int64),
        "k_us": torch.randn(8, 8, 2),
        "i_gt": torch.randn(8, 8, 2),
        "k_gt": torch.randn(8, 8, 2),
        "selected_mask": torch.zeros(8, 8, 2),
        "LR_pos": lr_pos,
        "LR_pos_norm": lr_pos_norm,
    }

    collator = KSpaceCollator(max_seq_len=16)
    batch = collator([sample, sample])

    assert "LR_i_gt" not in batch
    assert "LR_k_gt" not in batch
    assert batch["sampled_k"].shape == (2, 12, 2)
    assert batch["LR_pos_norm"].shape == (2, 16, 2)

