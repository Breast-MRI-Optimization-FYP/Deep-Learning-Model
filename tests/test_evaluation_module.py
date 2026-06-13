from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cli.evaluate import main_evaluate
from evaluation.acceleration import build_mask_metadata
from evaluation.metrics import per_sample_nmse
from model import KSpaceTransformer


def test_per_sample_nmse_matches_definition() -> None:
    gt = np.zeros((1, 2, 2, 2), dtype=np.float32)
    gt[..., 0] = 2.0
    pred = np.zeros_like(gt)
    pred[..., 0] = 1.0

    # Four pixels: error energy=4 and target energy=16.
    assert per_sample_nmse(pred, gt)[0] == pytest.approx(0.25)


def test_mask_metadata_uses_manifest_and_actual_acceleration(tmp_path: Path) -> None:
    masks = np.zeros((2, 8, 8), dtype=np.uint8)
    masks[0, ::2, :] = 1
    masks[1, ::4, :] = 1
    manifest = tmp_path / "masks.json"
    manifest.write_text(
        json.dumps(
            {
                "masks": [
                    {"mask_id": 0, "acceleration": 2, "sampling_pattern": "cartesian"},
                    {"mask_id": 1, "acceleration": 4, "sampling_pattern": "cartesian"},
                ]
            }
        ),
        encoding="utf-8",
    )

    metadata = build_mask_metadata(masks, manifest_path=manifest, requested_accelerations=(2, 4))
    assert [item.acceleration_label for item in metadata] == ["x2", "x4"]
    assert metadata[0].actual_acceleration == pytest.approx(2.0)
    assert metadata[1].actual_acceleration == pytest.approx(4.0)


def test_evaluate_cli_writes_complete_artifacts(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    hr = rng.normal(size=(1, 8, 8, 2)).astype(np.float32)
    masks = np.zeros((2, 8, 8), dtype=np.uint8)
    masks[0, ::2, :] = 1
    masks[1, ::4, :] = 1
    hr_path = tmp_path / "test_hr.npy"
    mask_path = tmp_path / "masks.npy"
    manifest_path = tmp_path / "mask_manifest.json"
    checkpoint_path = tmp_path / "checkpoint.pth"
    run_dir = tmp_path / "run"
    np.save(hr_path, hr)
    np.save(mask_path, masks)
    manifest_path.write_text(
        json.dumps(
            [
                {"mask_id": 0, "acceleration": 2, "sampling_pattern": "cartesian"},
                {"mask_id": 1, "acceleration": 4, "sampling_pattern": "cartesian"},
            ]
        ),
        encoding="utf-8",
    )

    model = KSpaceTransformer(
        lr_size=4,
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        num_lrdecoder_layers=1,
        num_hrdecoder_layers=1,
        dim_feedforward=32,
        hr_conv_channel=8,
        hr_conv_num=1,
        hr_kernel_size=3,
        dropout=0.0,
    )
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    exit_code = main_evaluate(
        [
            "--output_dir",
            str(run_dir),
            "--checkpoint",
            str(checkpoint_path),
            "--test_hr_data_path",
            str(hr_path),
            "--test_mask_path",
            str(mask_path),
            "--mask_manifest",
            str(manifest_path),
            "--acceleration_factors",
            "2",
            "4",
            "--batch_size",
            "1",
            "--lr_size",
            "4",
            "--max_seq_len",
            "64",
            "--num_workers",
            "0",
            "--d_model",
            "16",
            "--n_head",
            "4",
            "--num_encoder_layers",
            "1",
            "--num_LRdecoder_layers",
            "1",
            "--num_HRdecoder_layers",
            "1",
            "--dim_feedforward",
            "32",
            "--hr_conv_channel",
            "8",
            "--hr_conv_num",
            "1",
            "--hr_kernel_size",
            "3",
        ]
    )
    assert exit_code == 0

    evaluation_dir = run_dir / "evaluation"
    assert (evaluation_dir / "evaluation_summary.json").exists()
    assert (evaluation_dir / "per_sample_metrics.csv").exists()
    assert (evaluation_dir / "metrics_by_acceleration.csv").exists()
    assert (evaluation_dir / "comparison_table.md").exists()
    assert (evaluation_dir / "plots" / "nmse_by_acceleration.png").exists()
    assert len(list((evaluation_dir / "qualitative").glob("*.png"))) == 2

    summary = json.loads((evaluation_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert summary["num_records"] == 4
    assert set(summary["by_stage_and_acceleration"]) == {"K", "RM"}
    assert "nmse" in summary["by_stage_and_acceleration"]["RM"]["x2"]
