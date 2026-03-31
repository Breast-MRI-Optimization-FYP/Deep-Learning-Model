from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from kspace_transformer.config.cli import add_split_args, parse_runtime_config
from kspace_transformer.data.split import split_data_memory_efficient
from kspace_transformer.training.logger import RunLogger
from kspace_transformer.utils.seed import set_global_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split dataset indices for K-Space Transformer")
    add_split_args(parser)
    return parser


def main_split(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = parse_runtime_config(args)

    run_logger = RunLogger(prefix="split")
    run_logger.info("Runtime configuration validated.")

    hr_path = Path(args.hr_kspace_path)
    lr_path = Path(args.lr_kspace_path)
    if not hr_path.exists():
        raise FileNotFoundError(f"hr_kspace_path does not exist: {hr_path}")
    if not lr_path.exists():
        raise FileNotFoundError(f"lr_kspace_path does not exist: {lr_path}")

    set_global_seed(int(args.random_seed), deterministic=True)
    output_dir = Path(args.split_output_dir)

    indices = split_data_memory_efficient(
        k_data_path=hr_path,
        lr_k_data_path=lr_path,
        output_dir=output_dir,
        train_ratio=float(args.train_ratio),
        valid_ratio=float(args.valid_ratio),
        test_ratio=float(args.test_ratio),
        shuffle=bool(args.shuffle),
        random_seed=int(args.random_seed),
    )

    run_logger.info(
        f"Split complete. train={indices.train.shape[0]} valid={indices.valid.shape[0]} "
        f"test={indices.test.shape[0]} output_dir={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_split())
