from __future__ import annotations

import argparse
from pathlib import Path

from .defaults import build_default_config
from .schema import (
    DataConfig,
    EvalConfig,
    ModelConfig,
    PathConfig,
    RuntimeConfig,
    RuntimeOptions,
    TrainConfig,
    validate_runtime_config,
)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(
        f"Cannot parse boolean value from {value!r}. Use true/false style values."
    )


def _add_common_runtime_args(parser: argparse.ArgumentParser, defaults: RuntimeConfig) -> None:
    parser.add_argument("--output_dir", type=str, default=str(defaults.paths.output_dir))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume_train", type=str2bool, default=defaults.runtime.resume_train)
    parser.add_argument("--gpu", type=str, default=defaults.runtime.gpu)
    parser.add_argument("--seed", type=int, default=defaults.runtime.seed)


def _add_model_args(parser: argparse.ArgumentParser, defaults: RuntimeConfig) -> None:
    parser.add_argument("--d_model", type=int, default=defaults.model.d_model)
    parser.add_argument("--n_head", type=int, default=defaults.model.n_head)
    parser.add_argument("--num_encoder_layers", type=int, default=defaults.model.num_encoder_layers)
    parser.add_argument("--num_LRdecoder_layers", type=int, default=defaults.model.num_lrdecoder_layers)
    parser.add_argument("--num_HRdecoder_layers", type=int, default=defaults.model.num_hrdecoder_layers)
    parser.add_argument("--dim_feedforward", type=int, default=defaults.model.dim_feedforward)
    parser.add_argument("--hr_conv_channel", type=int, default=defaults.model.hr_conv_channel)
    parser.add_argument("--hr_conv_num", type=int, default=defaults.model.hr_conv_num)
    parser.add_argument("--hr_kernel_size", type=int, default=defaults.model.hr_kernel_size)


def _add_data_args(parser: argparse.ArgumentParser, defaults: RuntimeConfig) -> None:
    parser.add_argument("--batch_size", type=int, default=defaults.data.batch_size)
    parser.add_argument("--valid_batch_size", type=int, default=defaults.data.valid_batch_size)
    parser.add_argument("--lr_size", type=int, default=defaults.data.lr_size)
    parser.add_argument("--max_seq_len", type=int, default=defaults.data.max_seq_len)
    parser.add_argument("--reassign_mask", type=int, default=defaults.data.reassign_mask_every)


def add_train_args(parser: argparse.ArgumentParser) -> None:
    defaults = build_default_config()
    _add_common_runtime_args(parser, defaults)
    _add_model_args(parser, defaults)
    _add_data_args(parser, defaults)

    parser.add_argument("--epoch_num", type=int, default=defaults.train.epoch_num)
    parser.add_argument("--lr", type=float, default=defaults.train.lr)
    parser.add_argument("--l2norm", type=float, default=defaults.train.l2norm)
    parser.add_argument("--dropout", type=float, default=defaults.train.dropout)
    parser.add_argument("--kspace_loss", type=str2bool, default=defaults.train.kspace_loss)
    parser.add_argument("--img_loss", type=str2bool, default=defaults.train.img_loss)
    parser.add_argument("--lr_weights", nargs="+", type=float, default=list(defaults.train.lr_weights))
    parser.add_argument("--hr_weights", nargs="+", type=float, default=list(defaults.train.hr_weights))
    parser.add_argument("--conv_weight", type=float, default=defaults.train.conv_weight)
    parser.add_argument(
        "--pure_lr_training_epoch", type=int, default=defaults.train.pure_lr_training_epoch
    )
    parser.add_argument(
        "--pure_k_training_epoch", type=int, default=defaults.train.pure_k_training_epoch
    )

    parser.add_argument("--clear_memory", type=str2bool, default=defaults.runtime.clear_memory)
    parser.add_argument("--test_first_epoch", type=str2bool, default=defaults.runtime.test_first_epoch)
    parser.add_argument("--eval_interval_lr", type=int, default=defaults.eval.eval_interval_lr)
    parser.add_argument("--eval_interval_k", type=int, default=defaults.eval.eval_interval_k)
    parser.add_argument("--eval_interval_rm", type=int, default=defaults.eval.eval_interval_rm)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--train_hr_data_path", type=str, default=None)
    parser.add_argument("--train_lr_data_path", type=str, default=None)
    parser.add_argument("--train_mask_path", type=str, default=None)
    parser.add_argument("--valid_hr_data_path", type=str, default=None)
    parser.add_argument("--valid_lr_data_path", type=str, default=None)
    parser.add_argument("--valid_mask_path", type=str, default=None)


def add_test_args(parser: argparse.ArgumentParser) -> None:
    defaults = build_default_config()
    _add_common_runtime_args(parser, defaults)
    _add_model_args(parser, defaults)
    parser.add_argument("--batch_size", type=int, default=defaults.data.valid_batch_size)
    parser.add_argument("--lr_size", type=int, default=defaults.data.lr_size)
    parser.add_argument("--max_seq_len", type=int, default=defaults.data.max_seq_len)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--inference_stage", choices=["LR", "K", "RM"], default="RM")
    parser.add_argument("--up_scale", type=int, default=2)
    parser.add_argument("--save_summary_path", type=str, default=None)
    parser.add_argument("--test_hr_data_path", type=str, default=None)
    parser.add_argument("--test_lr_data_path", type=str, default=None)
    parser.add_argument("--test_mask_path", type=str, default=None)


def add_preprocess_args(parser: argparse.ArgumentParser) -> None:
    defaults = build_default_config()
    _add_common_runtime_args(parser, defaults)
    parser.add_argument("--input_hr_kspace_path", type=str, required=True)
    parser.add_argument("--output_lr_kspace_path", type=str, required=True)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=50)


def add_split_args(parser: argparse.ArgumentParser) -> None:
    defaults = build_default_config()
    _add_common_runtime_args(parser, defaults)
    parser.add_argument("--hr_kspace_path", type=str, required=True)
    parser.add_argument("--lr_kspace_path", type=str, required=True)
    parser.add_argument("--split_output_dir", type=str, required=True)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--shuffle", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=int, default=42)


def _get(namespace: argparse.Namespace, key: str, fallback: object) -> object:
    value = getattr(namespace, key, None)
    return fallback if value is None else value


def _path_or_none(value: object) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def parse_runtime_config(
    args: argparse.Namespace,
    *,
    require_paths: bool = False,
) -> RuntimeConfig:
    defaults = build_default_config()

    paths = PathConfig(
        output_dir=Path(str(_get(args, "output_dir", defaults.paths.output_dir))),
        checkpoint=_path_or_none(_get(args, "checkpoint", defaults.paths.checkpoint)),
        train_hr_data_path=_path_or_none(_get(args, "train_hr_data_path", None)),
        train_lr_data_path=_path_or_none(_get(args, "train_lr_data_path", None)),
        train_mask_path=_path_or_none(_get(args, "train_mask_path", None)),
        valid_hr_data_path=_path_or_none(_get(args, "valid_hr_data_path", None)),
        valid_lr_data_path=_path_or_none(_get(args, "valid_lr_data_path", None)),
        valid_mask_path=_path_or_none(_get(args, "valid_mask_path", None)),
        test_hr_data_path=_path_or_none(_get(args, "test_hr_data_path", None)),
        test_lr_data_path=_path_or_none(_get(args, "test_lr_data_path", None)),
        test_mask_path=_path_or_none(_get(args, "test_mask_path", None)),
    )

    data = DataConfig(
        batch_size=int(_get(args, "batch_size", defaults.data.batch_size)),
        valid_batch_size=int(_get(args, "valid_batch_size", defaults.data.valid_batch_size)),
        lr_size=int(_get(args, "lr_size", defaults.data.lr_size)),
        max_seq_len=int(_get(args, "max_seq_len", defaults.data.max_seq_len)),
        reassign_mask_every=int(_get(args, "reassign_mask", defaults.data.reassign_mask_every)),
    )

    model = ModelConfig(
        d_model=int(_get(args, "d_model", defaults.model.d_model)),
        n_head=int(_get(args, "n_head", defaults.model.n_head)),
        num_encoder_layers=int(_get(args, "num_encoder_layers", defaults.model.num_encoder_layers)),
        num_lrdecoder_layers=int(
            _get(args, "num_LRdecoder_layers", defaults.model.num_lrdecoder_layers)
        ),
        num_hrdecoder_layers=int(
            _get(args, "num_HRdecoder_layers", defaults.model.num_hrdecoder_layers)
        ),
        dim_feedforward=int(_get(args, "dim_feedforward", defaults.model.dim_feedforward)),
        hr_conv_channel=int(_get(args, "hr_conv_channel", defaults.model.hr_conv_channel)),
        hr_conv_num=int(_get(args, "hr_conv_num", defaults.model.hr_conv_num)),
        hr_kernel_size=int(_get(args, "hr_kernel_size", defaults.model.hr_kernel_size)),
    )

    train = TrainConfig(
        epoch_num=int(_get(args, "epoch_num", defaults.train.epoch_num)),
        lr=float(_get(args, "lr", defaults.train.lr)),
        l2norm=float(_get(args, "l2norm", defaults.train.l2norm)),
        dropout=float(_get(args, "dropout", defaults.train.dropout)),
        kspace_loss=bool(_get(args, "kspace_loss", defaults.train.kspace_loss)),
        img_loss=bool(_get(args, "img_loss", defaults.train.img_loss)),
        lr_weights=tuple(float(v) for v in _get(args, "lr_weights", defaults.train.lr_weights)),
        hr_weights=tuple(float(v) for v in _get(args, "hr_weights", defaults.train.hr_weights)),
        conv_weight=float(_get(args, "conv_weight", defaults.train.conv_weight)),
        pure_lr_training_epoch=int(
            _get(args, "pure_lr_training_epoch", defaults.train.pure_lr_training_epoch)
        ),
        pure_k_training_epoch=int(
            _get(args, "pure_k_training_epoch", defaults.train.pure_k_training_epoch)
        ),
    )

    runtime = RuntimeOptions(
        seed=int(_get(args, "seed", defaults.runtime.seed)),
        gpu=str(_get(args, "gpu", defaults.runtime.gpu)),
        resume_train=bool(_get(args, "resume_train", defaults.runtime.resume_train)),
        clear_memory=bool(_get(args, "clear_memory", defaults.runtime.clear_memory)),
        test_first_epoch=bool(_get(args, "test_first_epoch", defaults.runtime.test_first_epoch)),
    )

    eval_cfg = EvalConfig(
        eval_interval_lr=int(_get(args, "eval_interval_lr", defaults.eval.eval_interval_lr)),
        eval_interval_k=int(_get(args, "eval_interval_k", defaults.eval.eval_interval_k)),
        eval_interval_rm=int(_get(args, "eval_interval_rm", defaults.eval.eval_interval_rm)),
    )

    config = RuntimeConfig(
        paths=paths,
        data=data,
        model=model,
        train=train,
        runtime=runtime,
        eval=eval_cfg,
    )
    return validate_runtime_config(config, check_paths=require_paths)
