from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PathConfig:
    output_dir: Path = Path("Log/default")
    checkpoint: Path | None = None
    train_hr_data_path: Path | None = None
    train_lr_data_path: Path | None = None
    train_mask_path: Path | None = None
    valid_hr_data_path: Path | None = None
    valid_lr_data_path: Path | None = None
    valid_mask_path: Path | None = None
    test_hr_data_path: Path | None = None
    test_lr_data_path: Path | None = None
    test_mask_path: Path | None = None


@dataclass(frozen=True, slots=True)
class DataConfig:
    batch_size: int = 4
    valid_batch_size: int = 16
    lr_size: int = 64
    max_seq_len: int = 8000
    reassign_mask_every: int = 1


@dataclass(frozen=True, slots=True)
class ModelConfig:
    d_model: int = 256
    n_head: int = 4
    num_encoder_layers: int = 4
    num_lrdecoder_layers: int = 4
    num_hrdecoder_layers: int = 6
    dim_feedforward: int = 1024
    hr_conv_channel: int = 64
    hr_conv_num: int = 3
    hr_kernel_size: int = 3


@dataclass(frozen=True, slots=True)
class TrainConfig:
    epoch_num: int = 200
    lr: float = 5e-4
    l2norm: float = 0.0
    dropout: float = 0.0
    kspace_loss: bool = True
    img_loss: bool = True
    lr_weights: tuple[float, ...] = (0.3, 0.3, 0.3, 0.3)
    hr_weights: tuple[float, ...] = (0.3, 0.3, 0.3, 0.3, 0.3, 1.0)
    conv_weight: float = 1.0
    pure_lr_training_epoch: int = 50
    pure_k_training_epoch: int = 100


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    seed: int = 42
    gpu: str = "0"
    resume_train: bool = False
    clear_memory: bool = False
    test_first_epoch: bool = False


@dataclass(frozen=True, slots=True)
class EvalConfig:
    eval_interval_lr: int = 10
    eval_interval_k: int = 5
    eval_interval_rm: int = 1


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    paths: PathConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    runtime: RuntimeOptions
    eval: EvalConfig


def _is_strict_int(value: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _assert_positive_int(name: str, value: int) -> None:
    if not _is_strict_int(value) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _assert_non_negative_int(name: str, value: int) -> None:
    if not _is_strict_int(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


def _assert_positive_float(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value!r}")


def _assert_non_negative_float(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _validate_weight_vector(name: str, weights: tuple[float, ...], expected_len: int) -> None:
    if len(weights) != expected_len:
        raise ValueError(
            f"{name} length ({len(weights)}) must equal expected layers ({expected_len})"
        )
    for idx, weight in enumerate(weights):
        _assert_non_negative_float(f"{name}[{idx}]", float(weight))


def _validate_paths(paths: PathConfig, check_paths: bool) -> None:
    if not str(paths.output_dir).strip():
        raise ValueError("output_dir cannot be empty")

    if not check_paths:
        return

    required = {
        "train_hr_data_path": paths.train_hr_data_path,
        "train_lr_data_path": paths.train_lr_data_path,
        "train_mask_path": paths.train_mask_path,
        "valid_hr_data_path": paths.valid_hr_data_path,
        "valid_lr_data_path": paths.valid_lr_data_path,
        "valid_mask_path": paths.valid_mask_path,
    }
    for name, value in required.items():
        if value is None:
            raise ValueError(f"{name} is required when check_paths=True")
        if not value.exists():
            raise FileNotFoundError(f"{name} does not exist: {value}")

    if paths.checkpoint is not None and not paths.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {paths.checkpoint}")


def validate_runtime_config(config: RuntimeConfig, check_paths: bool = False) -> RuntimeConfig:
    _validate_paths(config.paths, check_paths=check_paths)

    _assert_positive_int("data.batch_size", config.data.batch_size)
    _assert_positive_int("data.valid_batch_size", config.data.valid_batch_size)
    _assert_positive_int("data.lr_size", config.data.lr_size)
    _assert_positive_int("data.max_seq_len", config.data.max_seq_len)
    _assert_positive_int("data.reassign_mask_every", config.data.reassign_mask_every)

    _assert_positive_int("model.d_model", config.model.d_model)
    _assert_positive_int("model.n_head", config.model.n_head)
    _assert_positive_int("model.num_encoder_layers", config.model.num_encoder_layers)
    _assert_positive_int("model.num_lrdecoder_layers", config.model.num_lrdecoder_layers)
    _assert_positive_int("model.num_hrdecoder_layers", config.model.num_hrdecoder_layers)
    _assert_positive_int("model.dim_feedforward", config.model.dim_feedforward)
    _assert_positive_int("model.hr_conv_channel", config.model.hr_conv_channel)
    _assert_positive_int("model.hr_conv_num", config.model.hr_conv_num)
    _assert_positive_int("model.hr_kernel_size", config.model.hr_kernel_size)

    _assert_positive_int("train.epoch_num", config.train.epoch_num)
    _assert_positive_float("train.lr", config.train.lr)
    _assert_non_negative_float("train.l2norm", config.train.l2norm)
    _assert_non_negative_float("train.dropout", config.train.dropout)

    if not _is_strict_int(config.train.pure_lr_training_epoch):
        raise TypeError(
            "train.pure_lr_training_epoch must be an int and not a bool"
        )
    if not _is_strict_int(config.train.pure_k_training_epoch):
        raise TypeError(
            "train.pure_k_training_epoch must be an int and not a bool"
        )

    _assert_non_negative_int(
        "train.pure_lr_training_epoch", config.train.pure_lr_training_epoch
    )
    _assert_non_negative_int(
        "train.pure_k_training_epoch", config.train.pure_k_training_epoch
    )

    if config.train.pure_lr_training_epoch >= config.train.pure_k_training_epoch:
        raise ValueError(
            "train.pure_lr_training_epoch must be smaller than train.pure_k_training_epoch"
        )

    if config.train.pure_k_training_epoch > config.train.epoch_num:
        raise ValueError(
            "train.pure_k_training_epoch cannot be larger than train.epoch_num"
        )

    _validate_weight_vector(
        "train.lr_weights", config.train.lr_weights, config.model.num_lrdecoder_layers
    )
    _validate_weight_vector(
        "train.hr_weights", config.train.hr_weights, config.model.num_hrdecoder_layers
    )
    _assert_non_negative_float("train.conv_weight", config.train.conv_weight)

    _assert_positive_int("runtime.seed", config.runtime.seed)

    _assert_positive_int("eval.eval_interval_lr", config.eval.eval_interval_lr)
    _assert_positive_int("eval.eval_interval_k", config.eval.eval_interval_k)
    _assert_positive_int("eval.eval_interval_rm", config.eval.eval_interval_rm)

    return config
