from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def _acceleration_value(label: str) -> float:
    try:
        return float(label.lower().removeprefix("x"))
    except ValueError:
        return float("inf")


def save_metric_plots(summary: dict[str, Any], output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grouped = summary["by_stage_and_acceleration"]
    stages = sorted(grouped)
    labels = sorted(
        {label for stage in stages for label in grouped[stage]},
        key=_acceleration_value,
    )
    if not stages or not labels:
        return []

    colors = {"K": "#2b6cb0", "RM": "#169c7b"}
    saved: list[Path] = []
    for metric in ("psnr", "ssim", "nmse"):
        series = {
            stage: [
                float(grouped[stage][label][metric]["mean"])
                if label in grouped[stage]
                else float("nan")
                for label in labels
            ]
            for stage in stages
        }
        finite = [value for values in series.values() for value in values if value == value]
        if not finite:
            continue
        minimum, maximum = min(finite), max(finite)
        if maximum <= minimum:
            maximum = minimum + 1.0

        width, height = 900, 520
        left, top, right, bottom = 90, 60, 40, 80
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((left, 20), f"{metric.upper()} by acceleration factor", fill="black")
        draw.line((left, top, left, height - bottom), fill="#555555", width=2)
        draw.line((left, height - bottom, width - right, height - bottom), fill="#555555", width=2)

        x_span = width - left - right
        y_span = height - top - bottom
        points_by_stage: dict[str, list[tuple[float, float]]] = {}
        for stage_index, stage in enumerate(stages):
            points: list[tuple[float, float]] = []
            for index, value in enumerate(series[stage]):
                if value != value:
                    continue
                x = left + (x_span / max(len(labels) - 1, 1)) * index
                y = top + (maximum - value) / (maximum - minimum) * y_span
                points.append((x, y))
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colors.get(stage, "#555555"))
            if len(points) > 1:
                draw.line(points, fill=colors.get(stage, "#555555"), width=3)
            points_by_stage[stage] = points
            legend_x = width - right - 120
            legend_y = 20 + stage_index * 20
            draw.line(
                (legend_x, legend_y + 7, legend_x + 24, legend_y + 7),
                fill=colors.get(stage, "#555555"),
                width=3,
            )
            draw.text((legend_x + 32, legend_y), stage, fill="black")

        for index, label in enumerate(labels):
            x = left + (x_span / max(len(labels) - 1, 1)) * index
            draw.text((x - 12, height - bottom + 12), label, fill="black")
        draw.text((8, top), f"{maximum:.5g}", fill="black")
        draw.text((8, height - bottom - 10), f"{minimum:.5g}", fill="black")

        path = output / f"{metric}_by_acceleration.png"
        image.save(path)
        saved.append(path)
    return saved
