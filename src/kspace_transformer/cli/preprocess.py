from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from kspace_transformer.config.cli import add_preprocess_args, parse_runtime_config
from kspace_transformer.data.lr_generation import generate_lr_kspace_batches
from kspace_transformer.training.logger import RunLogger
from kspace_transformer.utils.seed import set_global_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess MRI data for K-Space Transformer")
    add_preprocess_args(parser)
    return parser


def main_preprocess(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = parse_runtime_config(args)

    run_logger = RunLogger(prefix="preprocess")
    run_logger.info("Runtime configuration validated.")
    set_global_seed(config.runtime.seed, deterministic=True)

    input_path = Path(args.input_hr_kspace_path)
    output_path = Path(args.output_lr_kspace_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input_hr_kspace_path does not exist: {input_path}")

    generated_path = generate_lr_kspace_batches(
        input_path=input_path,
        output_path=output_path,
        scale=int(args.scale),
        batch_size=int(args.batch_size),
    )
    run_logger.info(f"Generated LR k-space data at {generated_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_preprocess())
