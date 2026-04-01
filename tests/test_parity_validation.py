from __future__ import annotations

import json
from pathlib import Path

from cli.parity import main_parity
from validation.parity import compare_run_bundles


def _train_summary(
    *,
    psnr: float,
    ssim: float,
    runtime_seconds: float,
    peak_memory_bytes: int,
) -> dict[str, float | int]:
    return {
        "best_valid_psnr": psnr,
        "best_valid_ssim": ssim,
        "runtime_seconds": runtime_seconds,
        "peak_memory_bytes": peak_memory_bytes,
    }


def _inference_summary(
    *,
    psnr: float,
    ssim: float,
    runtime_seconds: float,
    peak_memory_bytes: int,
) -> dict[str, float | int]:
    return {
        "mean_psnr": psnr,
        "mean_ssim": ssim,
        "runtime_seconds": runtime_seconds,
        "peak_memory_bytes": peak_memory_bytes,
    }


def test_compare_run_bundles_passes_for_small_drifts() -> None:
    baseline = {
        "train": _train_summary(
            psnr=35.0,
            ssim=0.9200,
            runtime_seconds=100.0,
            peak_memory_bytes=1_000_000,
        ),
        "inference": _inference_summary(
            psnr=34.8,
            ssim=0.9100,
            runtime_seconds=20.0,
            peak_memory_bytes=800_000,
        ),
    }
    candidate = {
        "train": _train_summary(
            psnr=34.98,
            ssim=0.9195,
            runtime_seconds=104.0,
            peak_memory_bytes=1_060_000,
        ),
        "inference": _inference_summary(
            psnr=34.77,
            ssim=0.9095,
            runtime_seconds=20.8,
            peak_memory_bytes=840_000,
        ),
    }

    report = compare_run_bundles(baseline, candidate)

    assert report["passed"] is True
    assert len(report["phases"]) == 2
    assert all(phase["passed"] for phase in report["phases"])


def test_compare_run_bundles_fails_on_psnr_drop() -> None:
    baseline = {
        "train": _train_summary(
            psnr=35.0,
            ssim=0.92,
            runtime_seconds=100.0,
            peak_memory_bytes=1_000_000,
        )
    }
    candidate = {
        "train": _train_summary(
            psnr=34.8,
            ssim=0.9198,
            runtime_seconds=101.0,
            peak_memory_bytes=1_020_000,
        )
    }

    report = compare_run_bundles(baseline, candidate)

    assert report["passed"] is False
    train_phase = report["phases"][0]
    psnr_check = next(check for check in train_phase["checks"] if check["name"] == "psnr")
    assert psnr_check["passed"] is False
    assert "dropped" in psnr_check["reason"]


def test_compare_run_bundles_no_comparable_phases_fails() -> None:
    report = compare_run_bundles({"preprocess": {}}, {"split": {}})

    assert report["passed"] is False
    assert "no comparable phases" in report["reason"]


def test_compare_run_bundles_allows_missing_memory_when_both_absent() -> None:
    baseline = {
        "train": {
            "best_valid_psnr": 35.0,
            "best_valid_ssim": 0.92,
            "runtime_seconds": 100.0,
        }
    }
    candidate = {
        "train": {
            "best_valid_psnr": 34.99,
            "best_valid_ssim": 0.9198,
            "runtime_seconds": 103.0,
        }
    }

    report = compare_run_bundles(baseline, candidate)

    assert report["passed"] is True
    memory_check = next(
        check for check in report["phases"][0]["checks"] if check["name"] == "peak_memory_bytes"
    )
    assert memory_check["passed"] is True


def test_parity_cli_writes_pass_report(tmp_path: Path) -> None:
    baseline_train_path = tmp_path / "baseline_train.json"
    candidate_train_path = tmp_path / "candidate_train.json"
    output_report_path = tmp_path / "report_pass.json"

    baseline_train_path.write_text(
        json.dumps(
            _train_summary(
                psnr=35.0,
                ssim=0.92,
                runtime_seconds=100.0,
                peak_memory_bytes=1_000_000,
            )
        ),
        encoding="utf-8",
    )
    candidate_train_path.write_text(
        json.dumps(
            _train_summary(
                psnr=34.98,
                ssim=0.9198,
                runtime_seconds=103.0,
                peak_memory_bytes=1_040_000,
            )
        ),
        encoding="utf-8",
    )

    exit_code = main_parity(
        [
            "--baseline_train_summary",
            str(baseline_train_path),
            "--candidate_train_summary",
            str(candidate_train_path),
            "--output_report",
            str(output_report_path),
        ]
    )

    assert exit_code == 0
    report = json.loads(output_report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True


def test_parity_cli_returns_nonzero_on_failure(tmp_path: Path) -> None:
    baseline_train_path = tmp_path / "baseline_train.json"
    candidate_train_path = tmp_path / "candidate_train.json"
    output_report_path = tmp_path / "report_fail.json"

    baseline_train_path.write_text(
        json.dumps(
            _train_summary(
                psnr=35.0,
                ssim=0.92,
                runtime_seconds=100.0,
                peak_memory_bytes=1_000_000,
            )
        ),
        encoding="utf-8",
    )
    candidate_train_path.write_text(
        json.dumps(
            _train_summary(
                psnr=34.99,
                ssim=0.9199,
                runtime_seconds=120.0,
                peak_memory_bytes=1_200_000,
            )
        ),
        encoding="utf-8",
    )

    exit_code = main_parity(
        [
            "--baseline_train_summary",
            str(baseline_train_path),
            "--candidate_train_summary",
            str(candidate_train_path),
            "--output_report",
            str(output_report_path),
            "--runtime_drift_ratio",
            "0.05",
        ]
    )

    assert exit_code == 1
    report = json.loads(output_report_path.read_text(encoding="utf-8"))
    assert report["passed"] is False

