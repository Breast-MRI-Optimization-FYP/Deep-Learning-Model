from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrainingStage(str, Enum):
    LR = "LR"
    K = "K"
    RM = "RM"


@dataclass(frozen=True, slots=True)
class StageWeights:
    stage: TrainingStage
    conv_weight: float
    lr_weights: tuple[float, ...]
    hr_weights: tuple[float, ...]
    eval_interval: int


class StageScheduler:
    def __init__(
        self,
        *,
        pure_lr_training_epoch: int,
        pure_k_training_epoch: int,
        lr_weights: tuple[float, ...],
        hr_weights: tuple[float, ...],
        conv_weight: float,
        num_lr_layers: int,
        num_hr_layers: int,
        eval_interval_lr: int = 10,
        eval_interval_k: int = 5,
        eval_interval_rm: int = 1,
    ) -> None:
        self.pure_lr_training_epoch = pure_lr_training_epoch
        self.pure_k_training_epoch = pure_k_training_epoch
        self.lr_weights = lr_weights
        self.hr_weights = hr_weights
        self.conv_weight = conv_weight
        self.num_lr_layers = num_lr_layers
        self.num_hr_layers = num_hr_layers
        self.eval_interval_lr = eval_interval_lr
        self.eval_interval_k = eval_interval_k
        self.eval_interval_rm = eval_interval_rm
        self._validate()

    def _validate(self) -> None:
        if self.pure_lr_training_epoch < 0:
            raise ValueError("pure_lr_training_epoch must be non-negative")
        if self.pure_k_training_epoch < 0:
            raise ValueError("pure_k_training_epoch must be non-negative")
        if self.pure_lr_training_epoch >= self.pure_k_training_epoch:
            raise ValueError("pure_lr_training_epoch must be smaller than pure_k_training_epoch")

        if self.num_lr_layers <= 0:
            raise ValueError("num_lr_layers must be positive")
        if self.num_hr_layers <= 0:
            raise ValueError("num_hr_layers must be positive")

        if len(self.lr_weights) != self.num_lr_layers:
            raise ValueError("lr_weights length must match num_lr_layers")
        if len(self.hr_weights) != self.num_hr_layers:
            raise ValueError("hr_weights length must match num_hr_layers")

        if self.eval_interval_lr <= 0 or self.eval_interval_k <= 0 or self.eval_interval_rm <= 0:
            raise ValueError("all eval intervals must be positive")

    def get_stage(self, epoch: int) -> TrainingStage:
        if epoch <= self.pure_lr_training_epoch:
            return TrainingStage.LR
        if epoch <= self.pure_k_training_epoch:
            return TrainingStage.K
        return TrainingStage.RM

    def weights_for_epoch(self, epoch: int) -> StageWeights:
        stage = self.get_stage(epoch)
        if stage is TrainingStage.RM:
            return StageWeights(
                stage=stage,
                conv_weight=self.conv_weight,
                lr_weights=self.lr_weights,
                hr_weights=self.hr_weights,
                eval_interval=self.eval_interval_rm,
            )

        if stage is TrainingStage.K:
            return StageWeights(
                stage=stage,
                conv_weight=0.0,
                lr_weights=self.lr_weights,
                hr_weights=self.hr_weights,
                eval_interval=self.eval_interval_k,
            )

        lr_weights = list(self.lr_weights)
        lr_weights[-1] = 1.0
        return StageWeights(
            stage=stage,
            conv_weight=0.0,
            lr_weights=tuple(lr_weights),
            hr_weights=tuple(0.0 for _ in range(self.num_hr_layers)),
            eval_interval=self.eval_interval_lr,
        )
