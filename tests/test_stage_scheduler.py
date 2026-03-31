from __future__ import annotations

import pytest

from kspace_transformer.training.stage import StageScheduler, TrainingStage


def _build_scheduler() -> StageScheduler:
    return StageScheduler(
        pure_lr_training_epoch=50,
        pure_k_training_epoch=100,
        lr_weights=(0.3, 0.3, 0.3, 0.3),
        hr_weights=(0.3, 0.3, 0.3, 0.3, 0.3, 1.0),
        conv_weight=1.0,
        num_lr_layers=4,
        num_hr_layers=6,
    )


def test_scheduler_boundary_mapping() -> None:
    scheduler = _build_scheduler()
    assert scheduler.get_stage(1) is TrainingStage.LR
    assert scheduler.get_stage(50) is TrainingStage.LR
    assert scheduler.get_stage(51) is TrainingStage.K
    assert scheduler.get_stage(100) is TrainingStage.K
    assert scheduler.get_stage(101) is TrainingStage.RM


def test_lr_stage_uses_only_last_lr_weight_and_zero_hr_weights() -> None:
    scheduler = _build_scheduler()
    weights = scheduler.weights_for_epoch(25)
    assert weights.stage is TrainingStage.LR
    assert weights.conv_weight == 0.0
    assert weights.hr_weights == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert weights.lr_weights == (0.3, 0.3, 0.3, 1.0)
    assert weights.eval_interval == 10


def test_invalid_stage_thresholds_raise() -> None:
    with pytest.raises(ValueError):
        StageScheduler(
            pure_lr_training_epoch=100,
            pure_k_training_epoch=100,
            lr_weights=(0.3, 0.3, 0.3, 0.3),
            hr_weights=(0.3, 0.3, 0.3, 0.3, 0.3, 1.0),
            conv_weight=1.0,
            num_lr_layers=4,
            num_hr_layers=6,
        )
