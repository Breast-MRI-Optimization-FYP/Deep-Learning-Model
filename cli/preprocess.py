from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from config.cli import add_preprocess_args, parse_runtime_config
from data.lr_generation import generate_lr_kspace_batches
from training.logger import RunLogger
from utils.perf import RuntimeTracker
from utils.seed import set_global_seed


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
    runtime_tracker = RuntimeTracker()
    runtime_tracker.start()

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
    summary = {
        "input_hr_kspace_path": str(input_path),
        "output_lr_kspace_path": str(generated_path),
        "scale": int(args.scale),
        "batch_size": int(args.batch_size),
        "runtime_seconds": runtime_tracker.elapsed_seconds(),
        "peak_memory_bytes": runtime_tracker.peak_memory_bytes(),
    }

    summary_path_arg = getattr(args, "save_summary_path", None)
    if summary_path_arg:
        summary_path = Path(summary_path_arg)
    else:
        summary_path = generated_path.parent / f"{generated_path.stem}_preprocess_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    run_logger.info(f"Generated LR k-space data at {generated_path}")
    run_logger.info(f"Preprocess summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_preprocess())

