from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kspace_transformer.training.checkpoint import CheckpointManager


def test_checkpoint_save_and_load_roundtrip(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 2)
    manager = CheckpointManager(tmp_path)

    state = {"epoch": 1, "model_state_dict": model.state_dict()}
    checkpoint_path = manager.save_checkpoint(state, "checkpoint.pth")

    assert checkpoint_path.exists()
    loaded = manager.load_checkpoint(checkpoint_path)
    assert loaded["epoch"] == 1


def test_save_best_tracks_improvement(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    state = {"epoch": 1}

    improved_1, _ = manager.save_best(state, metric=0.8, mode="max", filename="best.pth")
    improved_2, _ = manager.save_best(state, metric=0.7, mode="max", filename="best.pth")
    improved_3, _ = manager.save_best(state, metric=0.9, mode="max", filename="best.pth")

    assert improved_1 is True
    assert improved_2 is False
    assert improved_3 is True


def test_validate_state_dict_strict_and_non_strict(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 2)
    manager = CheckpointManager(tmp_path)

    good_state = model.state_dict()
    missing, unexpected = manager.validate_state_dict(model, good_state, strict=True)
    assert missing == []
    assert unexpected == []

    bad_state = dict(good_state)
    bad_state["extra.weight"] = torch.randn(1)

    missing_ns, unexpected_ns = manager.validate_state_dict(model, bad_state, strict=False)
    assert missing_ns == []
    assert "extra.weight" in unexpected_ns

    with pytest.raises(RuntimeError):
        manager.validate_state_dict(model, bad_state, strict=True)


def test_load_model_state_non_strict_returns_incompatibilities(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 2)
    manager = CheckpointManager(tmp_path)

    checkpoint = {"model_state_dict": dict(model.state_dict())}
    checkpoint["model_state_dict"].pop("bias")

    missing, unexpected = manager.load_model_state(model, checkpoint, strict=False)
    assert "bias" in missing
    assert unexpected == []
