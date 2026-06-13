from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_baseline_metrics(path: str | Path) -> list[dict[str, Any]]:
    """Load external baseline rows with method, acceleration, PSNR, SSIM, and optional NMSE."""
    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows = payload.get("results", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("Baseline JSON must be a list or contain a 'results' list")
    elif source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError("Baseline metrics must be JSON or CSV")

    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "method": str(row["method"]),
                "acceleration": str(row["acceleration"]),
                "psnr": float(row["psnr"]),
                "ssim": float(row["ssim"]),
                "nmse": float(row["nmse"]) if row.get("nmse") not in {None, ""} else None,
            }
        )
    return normalized
