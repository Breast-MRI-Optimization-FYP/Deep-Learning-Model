from __future__ import annotations

import argparse
from collections.abc import Sequence

from kspace_transformer.config.cli import add_test_args, parse_runtime_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference for K-Space Transformer")
    add_test_args(parser)
    return parser


def main_test(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = parse_runtime_config(args)
    print("Test entrypoint initialized. Runtime configuration validated.")
    print(f"Checkpoint path: {config.paths.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_test())
