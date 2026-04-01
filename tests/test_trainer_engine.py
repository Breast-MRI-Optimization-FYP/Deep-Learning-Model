from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from data.datasets import KSpaceCollator
from training.engine import Trainer
from training.losses import LossComputer
from training.stage import StageScheduler, TrainingStage


class DummyDataset(Dataset):
    def __init__(self, size: int = 6) -> None:
        self.size = size
        self.lr_pos = torch.stack(
            torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij"), dim=-1
        ).reshape(-1, 2)
        self.lr_pos_norm = torch.rand(16, 2)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sampled_len = 10 + (index % 3)
        unsampled_len = 12 + (index % 4)
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


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        *,
        src: torch.Tensor,
        lr_pos: torch.Tensor,
        src_pos: torch.Tensor,
        hr_pos: torch.Tensor,
        k_us: torch.Tensor,
        unsampled_pos: torch.Tensor,
        up_scale: int,
        mask: torch.Tensor,
        conv_weight: float,
        stage: str,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        batch_size = src.shape[0]
        lr_side = int(lr_pos.shape[1] ** 0.5)
        hr_h, hr_w = k_us.shape[1], k_us.shape[2]

        lr_pred = self.scale * torch.ones((batch_size, lr_side, lr_side, 2), device=src.device)
        up_lr = self.scale * torch.ones((batch_size, hr_h, hr_w, 2), device=src.device)
        hr_unconv = self.scale * torch.ones((batch_size, hr_h, hr_w, 2), device=src.device)
        hr_conv = self.scale * torch.ones((batch_size, hr_h, hr_w, 2), device=src.device)
        return [lr_pred], up_lr, up_lr, [hr_unconv], [hr_conv]


def _build_trainer() -> tuple[Trainer, DataLoader[Any], DataLoader[Any]]:
    dataset = DummyDataset(size=8)
    collator = KSpaceCollator(max_seq_len=32)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collator)
    valid_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collator)

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    stage_scheduler = StageScheduler(
        pure_lr_training_epoch=1,
        pure_k_training_epoch=2,
        lr_weights=(1.0,),
        hr_weights=(1.0,),
        conv_weight=1.0,
        num_lr_layers=1,
        num_hr_layers=1,
        eval_interval_lr=10,
        eval_interval_k=10,
        eval_interval_rm=10,
    )
    loss_computer = LossComputer(kspace_loss=True, img_loss=True)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        stage_scheduler=stage_scheduler,
        loss_computer=loss_computer,
        device=torch.device("cpu"),
    )
    return trainer, train_loader, valid_loader


def test_trainer_stage_transitions() -> None:
    trainer, train_loader, _ = _build_trainer()

    e1 = trainer.train_epoch(train_loader, epoch=1)
    e2 = trainer.train_epoch(train_loader, epoch=2)
    e3 = trainer.train_epoch(train_loader, epoch=3)

    assert e1.stage is TrainingStage.LR
    assert e2.stage is TrainingStage.K
    assert e3.stage is TrainingStage.RM

    assert torch.isfinite(torch.tensor(e1.loss))
    assert torch.isfinite(torch.tensor(e2.loss))
    assert torch.isfinite(torch.tensor(e3.loss))


def test_fit_respects_validation_guard() -> None:
    trainer, train_loader, valid_loader = _build_trainer()

    history = trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        start_epoch=1,
        end_epoch=3,
    )

    assert len(history) == 3
    assert history[0]["valid"] is not None
    assert history[1]["valid"] is None
    assert history[2]["valid"] is None

