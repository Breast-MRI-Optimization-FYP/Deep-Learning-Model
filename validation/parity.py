from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ParityThresholds:
    psnr_drift_db: float = 0.05
    ssim_drift: float = 0.001
    runtime_drift_ratio: float = 0.05
    memory_drift_ratio: float = 0.10


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    baseline: float | int | None
    candidate: float | int | None
    drift: float | None
    allowed: float
    passed: bool
    reason: str


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _absolute_lower_bound_check(
    name: str,
    baseline: float | None,
    candidate: float | None,
    allowed_drop: float,
) -> CheckResult:
    if baseline is None or candidate is None:
        return CheckResult(
            name=name,
            baseline=baseline,
            candidate=candidate,
            drift=None,
            allowed=allowed_drop,
            passed=False,
            reason="missing baseline or candidate metric",
        )

    drift = baseline - candidate
    passed = drift <= allowed_drop
    reason = "ok" if passed else f"{name} dropped by {drift:.6f}, allowed {allowed_drop:.6f}"
    return CheckResult(
        name=name,
        baseline=baseline,
        candidate=candidate,
        drift=drift,
        allowed=allowed_drop,
        passed=passed,
        reason=reason,
    )


def _relative_upper_bound_check(
    name: str,
    baseline: float | None,
    candidate: float | None,
    allowed_ratio: float,
    *,
    allow_missing: bool = False,
) -> CheckResult:
    if allow_missing and baseline is None and candidate is None:
        return CheckResult(
            name=name,
            baseline=baseline,
            candidate=candidate,
            drift=0.0,
            allowed=allowed_ratio,
            passed=True,
            reason="metric unavailable in both baseline and candidate",
        )

    if baseline is None or candidate is None:
        return CheckResult(
            name=name,
            baseline=baseline,
            candidate=candidate,
            drift=None,
            allowed=allowed_ratio,
            passed=False,
            reason="missing baseline or candidate metric",
        )

    if baseline <= 0:
        return CheckResult(
            name=name,
            baseline=baseline,
            candidate=candidate,
            drift=None,
            allowed=allowed_ratio,
            passed=False,
            reason="baseline metric must be positive for relative drift check",
        )

    drift_ratio = (candidate - baseline) / baseline
    passed = drift_ratio <= allowed_ratio
    reason = "ok" if passed else f"{name} increased by {drift_ratio:.6f}, allowed {allowed_ratio:.6f}"
    return CheckResult(
        name=name,
        baseline=baseline,
        candidate=candidate,
        drift=drift_ratio,
        allowed=allowed_ratio,
        passed=passed,
        reason=reason,
    )


def _extract_train_metrics(summary: dict[str, Any]) -> dict[str, float | None]:
    return {
        "psnr": _to_float(summary.get("best_valid_psnr")),
        "ssim": _to_float(summary.get("best_valid_ssim")),
        "runtime_seconds": _to_float(summary.get("runtime_seconds")),
        "peak_memory_bytes": _to_float(summary.get("peak_memory_bytes")),
    }


def _extract_inference_metrics(summary: dict[str, Any]) -> dict[str, float | None]:
    return {
        "psnr": _to_float(summary.get("mean_psnr")),
        "ssim": _to_float(summary.get("mean_ssim")),
        "runtime_seconds": _to_float(summary.get("runtime_seconds")),
        "peak_memory_bytes": _to_float(summary.get("peak_memory_bytes")),
    }


def compare_phase(
    *,
    phase_name: str,
    baseline_metrics: dict[str, float | None],
    candidate_metrics: dict[str, float | None],
    thresholds: ParityThresholds,
) -> dict[str, Any]:
    checks = [
        _absolute_lower_bound_check(
            "psnr",
            baseline_metrics.get("psnr"),
            candidate_metrics.get("psnr"),
            thresholds.psnr_drift_db,
        ),
        _absolute_lower_bound_check(
            "ssim",
            baseline_metrics.get("ssim"),
            candidate_metrics.get("ssim"),
            thresholds.ssim_drift,
        ),
        _relative_upper_bound_check(
            "runtime_seconds",
            baseline_metrics.get("runtime_seconds"),
            candidate_metrics.get("runtime_seconds"),
            thresholds.runtime_drift_ratio,
        ),
        _relative_upper_bound_check(
            "peak_memory_bytes",
            baseline_metrics.get("peak_memory_bytes"),
            candidate_metrics.get("peak_memory_bytes"),
            thresholds.memory_drift_ratio,
            allow_missing=True,
        ),
    ]

    return {
        "phase": phase_name,
        "passed": all(check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
    }


def compare_run_bundles(
    baseline_bundle: dict[str, Any],
    candidate_bundle: dict[str, Any],
    *,
    thresholds: ParityThresholds | None = None,
) -> dict[str, Any]:
    th = thresholds or ParityThresholds()

    phase_reports: list[dict[str, Any]] = []

    if "train" in baseline_bundle and "train" in candidate_bundle:
        phase_reports.append(
            compare_phase(
                phase_name="train",
                baseline_metrics=_extract_train_metrics(baseline_bundle["train"]),
                candidate_metrics=_extract_train_metrics(candidate_bundle["train"]),
                thresholds=th,
            )
        )

    if "inference" in baseline_bundle and "inference" in candidate_bundle:
        phase_reports.append(
            compare_phase(
                phase_name="inference",
                baseline_metrics=_extract_inference_metrics(baseline_bundle["inference"]),
                candidate_metrics=_extract_inference_metrics(candidate_bundle["inference"]),
                thresholds=th,
            )
        )

    if not phase_reports:
        return {
            "passed": False,
            "thresholds": asdict(th),
            "phases": [],
            "reason": "no comparable phases found; expected train and/or inference summaries",
        }

    return {
        "passed": all(phase["passed"] for phase in phase_reports),
        "thresholds": asdict(th),
        "phases": phase_reports,
    }
