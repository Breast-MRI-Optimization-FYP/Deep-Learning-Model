from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from config.cli import add_split_args, parse_runtime_config
from data.split import split_data_memory_efficient
from training.logger import RunLogger
from utils.perf import RuntimeTracker
from utils.seed import set_global_seed


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
    runtime_tracker = RuntimeTracker()
    runtime_tracker.start()

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

    summary = {
        "hr_kspace_path": str(hr_path),
        "lr_kspace_path": str(lr_path),
        "output_dir": str(output_dir),
        "train_ratio": float(args.train_ratio),
        "valid_ratio": float(args.valid_ratio),
        "test_ratio": float(args.test_ratio),
        "shuffle": bool(args.shuffle),
        "random_seed": int(args.random_seed),
        "train_samples": int(indices.train.shape[0]),
        "valid_samples": int(indices.valid.shape[0]),
        "test_samples": int(indices.test.shape[0]),
        "runtime_seconds": runtime_tracker.elapsed_seconds(),
        "peak_memory_bytes": runtime_tracker.peak_memory_bytes(),
    }

    summary_path_arg = getattr(args, "save_summary_path", None)
    if summary_path_arg:
        summary_path = Path(summary_path_arg)
    else:
        summary_path = output_dir / "split_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    run_logger.info(
        f"Split complete. train={indices.train.shape[0]} valid={indices.valid.shape[0]} "
        f"test={indices.test.shape[0]} output_dir={output_dir}"
    )
    run_logger.info(f"Split summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_split())

