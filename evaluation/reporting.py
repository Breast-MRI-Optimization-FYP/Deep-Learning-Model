from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .aggregation import aggregate_records, records_as_dicts
from .plots import save_metric_plots
from .schemas import EvaluationResult, SampleMetricRecord


def _acceleration_value(label: str) -> float:
    try:
        return float(label.lower().removeprefix("x"))
    except ValueError:
        return float("inf")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write_per_sample_csv(path: Path, records: list[SampleMetricRecord]) -> None:
    rows = records_as_dicts(records)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_grouped_csv(path: Path, summary: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for stage, accelerations in summary["by_stage_and_acceleration"].items():
        for acceleration, metrics in accelerations.items():
            row: dict[str, Any] = {"stage": stage, "acceleration": acceleration}
            for metric, stats in metrics.items():
                for stat_name, value in stats.items():
                    row[f"{metric}_{stat_name}"] = value
            rows.append(row)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    summary: dict[str, Any],
    baseline_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Reconstruction Evaluation",
        "",
        "| Stage | Acceleration | PSNR mean | SSIM mean | NMSE mean | Samples |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    grouped = summary["by_stage_and_acceleration"]
    for stage in sorted(grouped):
        for acceleration in sorted(grouped[stage], key=_acceleration_value):
            metrics = grouped[stage][acceleration]
            lines.append(
                f"| {stage} | {acceleration} | {metrics['psnr']['mean']:.4f} | "
                f"{metrics['ssim']['mean']:.6f} | {metrics['nmse']['mean']:.8f} | "
                f"{metrics['psnr']['count']} |"
            )

    if summary["improvements"]:
        lines.extend(
            [
                "",
                "## Refinement Improvement Over K-Space Output",
                "",
                "| Acceleration | PSNR delta | SSIM delta | NMSE reduction |",
                "|---|---:|---:|---:|",
            ]
        )
        for acceleration, metrics in summary["improvements"].items():
            lines.append(
                f"| {acceleration} | {metrics['psnr']['mean']:+.4f} | "
                f"{metrics['ssim']['mean']:+.6f} | "
                f"{metrics['nmse_reduction']['mean']:+.8f} |"
            )
    if baseline_rows:
        lines.extend(
            [
                "",
                "## External Baselines",
                "",
                "| Method | Acceleration | PSNR | SSIM | NMSE |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in baseline_rows:
            nmse = "" if row.get("nmse") is None else f"{float(row['nmse']):.8f}"
            lines.append(
                f"| {row['method']} | {row['acceleration']} | {float(row['psnr']):.4f} | "
                f"{float(row['ssim']):.6f} | {nmse} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_evaluation_reports(
    result: EvaluationResult,
    output_dir: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    baseline_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = aggregate_records(result.records)
    preferred_stage = "RM" if "RM" in summary["overall"] else next(iter(summary["overall"]), None)
    preferred_metrics = summary["overall"].get(preferred_stage, {}) if preferred_stage else {}
    payload = {
        **(metadata or {}),
        "runtime_seconds": result.runtime_seconds,
        "peak_memory_bytes": result.peak_memory_bytes,
        "stage": preferred_stage,
        "mean_psnr": preferred_metrics.get("psnr", {}).get("mean"),
        "mean_ssim": preferred_metrics.get("ssim", {}).get("mean"),
        "mean_nmse": preferred_metrics.get("nmse", {}).get("mean"),
        **summary,
        "external_baselines": baseline_rows or [],
    }

    paths = {
        "summary": output / "evaluation_summary.json",
        "per_sample": output / "per_sample_metrics.csv",
        "grouped": output / "metrics_by_acceleration.csv",
        "markdown": output / "comparison_table.md",
    }
    paths["summary"].write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )
    _write_per_sample_csv(paths["per_sample"], result.records)
    _write_grouped_csv(paths["grouped"], summary)
    _write_markdown(paths["markdown"], summary, baseline_rows or [])
    for plot_path in save_metric_plots(summary, output / "plots"):
        paths[f"plot_{plot_path.stem}"] = plot_path
    return paths
