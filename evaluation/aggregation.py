from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from statistics import mean, median, pstdev
from typing import Any, Iterable

import numpy as np

from .schemas import SampleMetricRecord


def _metric_stats(values: list[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return {
            "count": len(values),
            "finite_count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": len(values),
        "finite_count": len(finite),
        "mean": mean(finite),
        "std": pstdev(finite) if len(finite) > 1 else 0.0,
        "median": median(finite),
        "min": min(finite),
        "max": max(finite),
    }


def aggregate_records(records: Iterable[SampleMetricRecord]) -> dict[str, Any]:
    records_list = list(records)
    grouped: dict[tuple[str, str], list[SampleMetricRecord]] = defaultdict(list)
    by_stage: dict[str, list[SampleMetricRecord]] = defaultdict(list)

    for record in records_list:
        grouped[(record.stage, record.acceleration_label)].append(record)
        by_stage[record.stage].append(record)

    def summarize(items: list[SampleMetricRecord]) -> dict[str, Any]:
        return {
            metric: _metric_stats([float(getattr(item, metric)) for item in items])
            for metric in ("psnr", "ssim", "nmse")
        }

    grouped_summary: dict[str, dict[str, Any]] = defaultdict(dict)
    for (stage, acceleration), items in sorted(grouped.items()):
        grouped_summary[stage][acceleration] = summarize(items)

    overall = {stage: summarize(items) for stage, items in sorted(by_stage.items())}
    return {
        "overall": overall,
        "by_stage_and_acceleration": dict(grouped_summary),
        "improvements": calculate_stage_improvements(records_list),
        "num_records": len(records_list),
    }


def calculate_stage_improvements(records: list[SampleMetricRecord]) -> dict[str, Any]:
    paired: dict[tuple[int, int, str], dict[str, SampleMetricRecord]] = defaultdict(
        dict
    )
    for record in records:
        key = (record.sample_index, record.mask_id, record.acceleration_label)
        paired[key][record.stage] = record

    deltas: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"psnr": [], "ssim": [], "nmse_reduction": []}
    )
    for (_, _, acceleration), stage_records in paired.items():
        if "K" not in stage_records or "RM" not in stage_records:
            continue
        k_record = stage_records["K"]
        rm_record = stage_records["RM"]
        deltas[acceleration]["psnr"].append(rm_record.psnr - k_record.psnr)
        deltas[acceleration]["ssim"].append(rm_record.ssim - k_record.ssim)
        deltas[acceleration]["nmse_reduction"].append(k_record.nmse - rm_record.nmse)

    return {
        acceleration: {
            metric: _metric_stats(values)
            for metric, values in metrics.items()
        }
        for acceleration, metrics in sorted(deltas.items())
    }


def records_as_dicts(records: Iterable[SampleMetricRecord]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]
