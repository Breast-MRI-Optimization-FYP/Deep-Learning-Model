from __future__ import annotations

import torch

from model import KSpaceTransformer
from model.positional_encoding import PositionalEncoding


def _build_inputs(
    batch_size: int = 2,
    h: int = 8,
    w: int = 8,
    lr_size: int = 4,
    src_len: int = 18,
    query_len: int = 22,
) -> dict[str, torch.Tensor]:
    src = torch.randn(batch_size, src_len, 2)
    src_pos = torch.rand(batch_size, src_len, 2)

    lr_pos = torch.rand(batch_size, lr_size * lr_size, 2)
    hr_pos = torch.rand(batch_size, query_len, 2)

    k_us = torch.randn(batch_size, h, w, 2)
    mask = torch.randint(0, 2, (batch_size, h, w, 2), dtype=torch.int64).to(torch.float32)

    unsampled_pos = torch.randint(0, h, (batch_size, query_len, 2), dtype=torch.int64)
    return {
        "src": src,
        "src_pos": src_pos,
        "lr_pos": lr_pos,
        "hr_pos": hr_pos,
        "k_us": k_us,
        "mask": mask,
        "unsampled_pos": unsampled_pos,
    }


def _build_model(lr_size: int = 4) -> KSpaceTransformer:
    return KSpaceTransformer(
        lr_size=lr_size,
        channel=2,
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        num_lrdecoder_layers=2,
        num_hrdecoder_layers=2,
        dim_feedforward=64,
        hr_conv_channel=16,
        hr_conv_num=1,
        hr_kernel_size=3,
        dropout=0.0,
        activation="relu",
    )


def test_positional_encoding_uses_input_device() -> None:
    pe = PositionalEncoding(pe_dim=32, magnify=250.0)
    pos = torch.rand(2, 10, 2)
    out = pe(pos)

    assert out.device == pos.device
    assert out.shape == (2, 10, 32)


def test_model_forward_lr_stage_outputs() -> None:
    inputs = _build_inputs()
    model = _build_model()

    outputs = model(
        src=inputs["src"],
        lr_pos=inputs["lr_pos"],
        src_pos=inputs["src_pos"],
        hr_pos=inputs["hr_pos"],
        k_us=inputs["k_us"],
        unsampled_pos=inputs["unsampled_pos"],
        up_scale=2,
        mask=inputs["mask"],
        conv_weight=1.0,
        stage="LR",
    )

    assert len(outputs.lr_images) == 2
    assert len(outputs.hr_transformer_images) == 2
    assert len(outputs.hr_refined_images) == 2
    assert outputs.up_lr_image is not None
    assert outputs.up_lr_image.shape == (2, 8, 8, 2)
    assert torch.allclose(outputs.hr_transformer_images[0], torch.zeros_like(outputs.hr_transformer_images[0]))


def test_model_forward_k_and_rm_stages() -> None:
    inputs = _build_inputs()
    model = _build_model()

    outputs_k = model(
        src=inputs["src"],
        lr_pos=inputs["lr_pos"],
        src_pos=inputs["src_pos"],
        hr_pos=inputs["hr_pos"],
        k_us=inputs["k_us"],
        unsampled_pos=inputs["unsampled_pos"],
        up_scale=2,
        mask=inputs["mask"],
        conv_weight=1.0,
        stage="K",
    )
    assert len(outputs_k.hr_transformer_images) == 2
    assert len(outputs_k.hr_refined_images) == 2
    assert torch.isfinite(outputs_k.hr_transformer_images[-1]).all()
    assert torch.allclose(outputs_k.hr_refined_images[-1], torch.zeros_like(outputs_k.hr_refined_images[-1]))

    outputs_rm = model(
        src=inputs["src"],
        lr_pos=inputs["lr_pos"],
        src_pos=inputs["src_pos"],
        hr_pos=inputs["hr_pos"],
        k_us=inputs["k_us"],
        unsampled_pos=inputs["unsampled_pos"],
        up_scale=2,
        mask=inputs["mask"],
        conv_weight=1.0,
        stage="RM",
    )
    assert len(outputs_rm.hr_refined_images) == 2
    assert torch.isfinite(outputs_rm.hr_refined_images[-1]).all()

