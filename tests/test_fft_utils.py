from __future__ import annotations

import torch

from utils.fftc import fft2c, ifft2c


def test_fft_ifft_roundtrip_close() -> None:
    data = torch.randn(2, 8, 8, 2, dtype=torch.float32)
    reconstructed = ifft2c(fft2c(data), need_shift=True)
    assert torch.allclose(data, reconstructed, atol=1e-5, rtol=1e-5)

