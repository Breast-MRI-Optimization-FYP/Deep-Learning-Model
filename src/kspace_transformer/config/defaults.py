from __future__ import annotations

from .schema import (
    DataConfig,
    EvalConfig,
    ModelConfig,
    PathConfig,
    RuntimeConfig,
    RuntimeOptions,
    TrainConfig,
)


def build_default_config() -> RuntimeConfig:
    return RuntimeConfig(
        paths=PathConfig(),
        data=DataConfig(),
        model=ModelConfig(),
        train=TrainConfig(),
        runtime=RuntimeOptions(),
        eval=EvalConfig(),
    )
