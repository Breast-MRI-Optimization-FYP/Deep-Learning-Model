from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from data.masks import ensure_mask_channels

from .schemas import MaskMetadata


def actual_acceleration(mask: np.ndarray) -> float:
    sampled = int(np.asarray(mask)[..., 0].astype(bool).sum())
    if sampled <= 0:
        raise ValueError("Mask must contain at least one sampled k-space position")
    height, width = mask.shape[:2]
    return float(height * width / sampled)


def _load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("masks", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("JSON mask manifest must be a list or contain a 'masks' list")
        return [dict(row) for row in rows]

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    raise ValueError("Mask manifest must be JSON or CSV")


def build_mask_metadata(
    mask_bank: np.ndarray,
    *,
    manifest_path: str | Path | None = None,
    requested_accelerations: tuple[float, ...] = (),
    matching_tolerance: float = 0.35,
) -> list[MaskMetadata]:
    masks = ensure_mask_channels(mask_bank)
    manifest_by_id: dict[int, dict[str, Any]] = {}
    if manifest_path is not None:
        for row in _load_manifest_rows(Path(manifest_path)):
            mask_id = int(row["mask_id"])
            manifest_by_id[mask_id] = row

    metadata: list[MaskMetadata] = []
    for mask_id, mask in enumerate(masks):
        actual = actual_acceleration(mask)
        row = manifest_by_id.get(mask_id)

        if row is not None:
            acceleration = float(row["acceleration"])
            pattern = str(row.get("sampling_pattern", row.get("pattern", "unknown")))
        elif requested_accelerations:
            acceleration = min(requested_accelerations, key=lambda value: abs(value - actual))
            relative_error = abs(acceleration - actual) / acceleration
            if relative_error > matching_tolerance:
                continue
            pattern = "unknown"
        else:
            acceleration = round(actual, 2)
            pattern = "unknown"

        if requested_accelerations and all(
            abs(acceleration - requested) > 1e-6 for requested in requested_accelerations
        ):
            continue

        metadata.append(
            MaskMetadata(
                mask_id=mask_id,
                acceleration=acceleration,
                sampling_pattern=pattern,
                actual_acceleration=actual,
            )
        )

    if not metadata:
        raise ValueError("No masks matched the requested evaluation acceleration factors")
    return metadata
