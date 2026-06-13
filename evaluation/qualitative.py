from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .schemas import QualitativeSample


def _normalize_uint8(image: np.ndarray, *, low: float, high: float) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    normalized = np.clip((array - low) / (high - low), 0.0, 1.0)
    return np.round(normalized * 255.0).astype(np.uint8)


def save_qualitative_samples(
    samples: list[QualitativeSample],
    output_dir: str | Path,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for sample in samples:
        panels: list[tuple[str, np.ndarray]] = [
            ("Undersampled", sample.undersampled),
            ("K-space output", sample.predictions.get("K", sample.undersampled)),
            (
                "Refined output",
                sample.predictions.get(
                    "RM",
                    sample.predictions.get("K", sample.undersampled),
                ),
            ),
            ("Ground truth", sample.ground_truth),
        ]
        combined = np.concatenate([np.asarray(image).reshape(-1) for _, image in panels])
        low, high = np.percentile(combined, [0.5, 99.5])
        panel_images = [
            Image.fromarray(_normalize_uint8(image, low=low, high=high), mode="L").convert("RGB")
            for _, image in panels
        ]
        width = max(image.width for image in panel_images)
        height = max(image.height for image in panel_images)
        title_height = 28
        canvas = Image.new("RGB", (width * 2, (height + title_height) * 2), "white")
        draw = ImageDraw.Draw(canvas)

        for panel_index, ((title, _), image) in enumerate(zip(panels, panel_images)):
            row, col = divmod(panel_index, 2)
            x = col * width
            y = row * (height + title_height)
            canvas.paste(image.resize((width, height)), (x, y + title_height))
            draw.text((x + 6, y + 6), title, fill="black")

        path = output / (
            f"{sample.acceleration_label}_sample_{sample.sample_index:04d}"
            f"_mask_{sample.mask_id:03d}.png"
        )
        canvas.save(path)
        saved.append(path)
    return saved
