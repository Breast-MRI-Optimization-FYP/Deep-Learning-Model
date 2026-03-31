from __future__ import annotations

import argparse
from collections.abc import Sequence

from kspace_transformer.config.cli import add_preprocess_args, parse_runtime_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess MRI data for K-Space Transformer")
    add_preprocess_args(parser)
    return parser


def main_preprocess(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = parse_runtime_config(args)
    print("Preprocess entrypoint initialized. Runtime configuration validated.")
    print(f"Output directory: {config.paths.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_preprocess())
