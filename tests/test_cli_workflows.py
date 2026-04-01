from __future__ import annotations

from pathlib import Path

import numpy as np

from cli.preprocess import main_preprocess
from cli.split import main_split
from cli.test import main_test
from cli.train import main_train


def _write_small_dataset(root: Path) -> dict[str, Path]:
    rng = np.random.default_rng(7)

    train_hr = rng.normal(size=(2, 8, 8, 2)).astype(np.float32)
    train_lr = rng.normal(size=(2, 4, 4, 2)).astype(np.float32)
    valid_hr = rng.normal(size=(2, 8, 8, 2)).astype(np.float32)
    valid_lr = rng.normal(size=(2, 4, 4, 2)).astype(np.float32)
    test_hr = rng.normal(size=(2, 8, 8, 2)).astype(np.float32)

    mask = np.zeros((3, 8, 8), dtype=np.uint8)
    mask[:, ::2, :] = 1
    mask[:, :, ::2] = 1

    paths = {
        "train_hr": root / "train_hr.npy",
        "train_lr": root / "train_lr.npy",
        "valid_hr": root / "valid_hr.npy",
        "valid_lr": root / "valid_lr.npy",
        "test_hr": root / "test_hr.npy",
        "mask": root / "mask.npy",
    }

    np.save(paths["train_hr"], train_hr)
    np.save(paths["train_lr"], train_lr)
    np.save(paths["valid_hr"], valid_hr)
    np.save(paths["valid_lr"], valid_lr)
    np.save(paths["test_hr"], test_hr)
    np.save(paths["mask"], mask)

    return paths


def test_preprocess_and_split_cli(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    hr = rng.normal(size=(10, 8, 8, 2)).astype(np.float32)

    hr_path = tmp_path / "hr.npy"
    lr_path = tmp_path / "lr_generated.npy"
    split_dir = tmp_path / "splits"

    np.save(hr_path, hr)

    assert (
        main_preprocess(
            [
                "--input_hr_kspace_path",
                str(hr_path),
                "--output_lr_kspace_path",
                str(lr_path),
                "--scale",
                "2",
                "--batch_size",
                "2",
            ]
        )
        == 0
    )
    assert lr_path.exists()

    assert (
        main_split(
            [
                "--hr_kspace_path",
                str(hr_path),
                "--lr_kspace_path",
                str(lr_path),
                "--split_output_dir",
                str(split_dir),
                "--train_ratio",
                "0.7",
                "--valid_ratio",
                "0.2",
                "--test_ratio",
                "0.1",
                "--shuffle",
                "false",
                "--random_seed",
                "42",
            ]
        )
        == 0
    )

    assert (split_dir / "train_k.npy").exists()
    assert (split_dir / "valid_k.npy").exists()
    assert (split_dir / "test_k.npy").exists()
    assert (split_dir / "train_lr_k.npy").exists()
    assert (split_dir / "valid_lr_k.npy").exists()
    assert (split_dir / "test_lr_k.npy").exists()


def test_train_and_test_cli_smoke(tmp_path: Path) -> None:
    paths = _write_small_dataset(tmp_path)
    run_dir = tmp_path / "run"

    train_exit = main_train(
        [
            "--output_dir",
            str(run_dir),
            "--epoch_num",
            "1",
            "--pure_lr_training_epoch",
            "0",
            "--pure_k_training_epoch",
            "1",
            "--batch_size",
            "1",
            "--valid_batch_size",
            "1",
            "--lr_size",
            "4",
            "--max_seq_len",
            "32",
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
            "--lr_weights",
            "1.0",
            "--hr_weights",
            "1.0",
            "--train_hr_data_path",
            str(paths["train_hr"]),
            "--train_lr_data_path",
            str(paths["train_lr"]),
            "--train_mask_path",
            str(paths["mask"]),
            "--valid_hr_data_path",
            str(paths["valid_hr"]),
            "--valid_lr_data_path",
            str(paths["valid_lr"]),
            "--valid_mask_path",
            str(paths["mask"]),
        ]
    )
    assert train_exit == 0

    best_ckpt = run_dir / "checkpoints" / "best_valid_psnr.pth"
    if not best_ckpt.exists():
        best_ckpt = run_dir / "checkpoints" / "last.pth"
    assert best_ckpt.exists()

    summary_path = tmp_path / "inference_summary.json"
    test_exit = main_test(
        [
            "--output_dir",
            str(run_dir),
            "--checkpoint",
            str(best_ckpt),
            "--test_hr_data_path",
            str(paths["test_hr"]),
            "--test_mask_path",
            str(paths["mask"]),
            "--batch_size",
            "1",
            "--lr_size",
            "4",
            "--max_seq_len",
            "32",
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
            "--inference_stage",
            "RM",
            "--save_summary_path",
            str(summary_path),
        ]
    )
    assert test_exit == 0
    assert summary_path.exists()

