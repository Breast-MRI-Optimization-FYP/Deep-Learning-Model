from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.logger import RunLogger
from validation.parity import ParityThresholds, compare_run_bundles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare baseline and candidate run summaries")

    parser.add_argument("--baseline_bundle", type=str, default=None)
    parser.add_argument("--candidate_bundle", type=str, default=None)

    parser.add_argument("--baseline_train_summary", type=str, default=None)
    parser.add_argument("--baseline_inference_summary", type=str, default=None)
    parser.add_argument("--candidate_train_summary", type=str, default=None)
    parser.add_argument("--candidate_inference_summary", type=str, default=None)

    parser.add_argument("--save_baseline_bundle", type=str, default=None)
    parser.add_argument("--save_candidate_bundle", type=str, default=None)

    parser.add_argument("--psnr_drift_db", type=float, default=0.05)
    parser.add_argument("--ssim_drift", type=float, default=0.001)
    parser.add_argument("--runtime_drift_ratio", type=float, default=0.05)
    parser.add_argument("--memory_drift_ratio", type=float, default=0.10)

    parser.add_argument("--output_report", type=str, required=True)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _build_bundle_from_summaries(train_path: str | None, inference_path: str | None) -> dict[str, Any]:
    bundle: dict[str, Any] = {}
    if train_path:
        bundle["train"] = _load_json(Path(train_path))
    if inference_path:
        bundle["inference"] = _load_json(Path(inference_path))
    return bundle


def _resolve_bundle(
    *,
    bundle_path: str | None,
    train_summary_path: str | None,
    inference_summary_path: str | None,
) -> dict[str, Any]:
    if bundle_path:
        return _load_json(Path(bundle_path))
    return _build_bundle_from_summaries(train_summary_path, inference_summary_path)


def main_parity(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_logger = RunLogger(prefix="parity")

    baseline_bundle = _resolve_bundle(
        bundle_path=args.baseline_bundle,
        train_summary_path=args.baseline_train_summary,
        inference_summary_path=args.baseline_inference_summary,
    )
    candidate_bundle = _resolve_bundle(
        bundle_path=args.candidate_bundle,
        train_summary_path=args.candidate_train_summary,
        inference_summary_path=args.candidate_inference_summary,
    )

    if args.save_baseline_bundle:
        baseline_bundle_path = Path(args.save_baseline_bundle)
        baseline_bundle_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_bundle_path.write_text(json.dumps(baseline_bundle, indent=2), encoding="utf-8")

    if args.save_candidate_bundle:
        candidate_bundle_path = Path(args.save_candidate_bundle)
        candidate_bundle_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_bundle_path.write_text(json.dumps(candidate_bundle, indent=2), encoding="utf-8")

    thresholds = ParityThresholds(
        psnr_drift_db=float(args.psnr_drift_db),
        ssim_drift=float(args.ssim_drift),
        runtime_drift_ratio=float(args.runtime_drift_ratio),
        memory_drift_ratio=float(args.memory_drift_ratio),
    )

    report = compare_run_bundles(
        baseline_bundle,
        candidate_bundle,
        thresholds=thresholds,
    )

    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    run_logger.info(f"Parity report saved to {output_path}")
    run_logger.info(f"Parity result: {'PASS' if report['passed'] else 'FAIL'}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main_parity())

