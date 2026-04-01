from __future__ import annotations

from dataclasses import replace

import pytest

from config.defaults import build_default_config
from config.schema import validate_runtime_config


def test_default_config_is_valid() -> None:
    cfg = build_default_config()
    validated = validate_runtime_config(cfg)
    assert validated == cfg


def test_invalid_epoch_order_raises() -> None:
    cfg = build_default_config()
    bad_cfg = replace(
        cfg,
        train=replace(cfg.train, pure_lr_training_epoch=120, pure_k_training_epoch=100),
    )
    with pytest.raises(ValueError):
        validate_runtime_config(bad_cfg)


def test_boolean_epoch_value_is_rejected() -> None:
    cfg = build_default_config()
    bad_cfg = replace(cfg, train=replace(cfg.train, pure_lr_training_epoch=True))
    with pytest.raises(TypeError):
        validate_runtime_config(bad_cfg)


def test_lr_weight_length_mismatch_raises() -> None:
    cfg = build_default_config()
    bad_cfg = replace(cfg, model=replace(cfg.model, num_lrdecoder_layers=3))
    with pytest.raises(ValueError):
        validate_runtime_config(bad_cfg)


def test_hr_weight_length_mismatch_raises() -> None:
    cfg = build_default_config()
    bad_cfg = replace(cfg, model=replace(cfg.model, num_hrdecoder_layers=5))
    with pytest.raises(ValueError):
        validate_runtime_config(bad_cfg)

