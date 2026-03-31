from __future__ import annotations

import argparse
from collections.abc import Sequence

from kspace_transformer.config.cli import add_train_args, parse_runtime_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train K-Space Transformer model")
    add_train_args(parser)
    return parser


def main_train(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = parse_runtime_config(args)
    print("Train entrypoint initialized. Runtime configuration validated.")
    print(
        "Stage boundaries: "
        f"LR<= {config.train.pure_lr_training_epoch}, "
        f"K<= {config.train.pure_k_training_epoch}, "
        "RM> pure_k_training_epoch"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_train())
