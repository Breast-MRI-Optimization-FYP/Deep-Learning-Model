from __future__ import annotations

from typing import Optional

import torch


def _validate_complex_tensor(data: torch.Tensor) -> None:
    if data.ndim < 3 or data.shape[-1] != 2:
        raise ValueError("Expected tensor with complex channel in last dim of size 2")


def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def roll(x: torch.Tensor, shift: list[int], dim: list[int]) -> torch.Tensor:
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    out = x
    for amount, axis in zip(shift, dim):
        out = roll_one_dim(out, amount, axis)
    return out


def fftshift(x: torch.Tensor, dim: Optional[list[int]] = None) -> torch.Tensor:
    if dim is None:
        dim = list(range(x.ndim))
    shift = [x.shape[d] // 2 for d in dim]
    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[list[int]] = None) -> torch.Tensor:
    if dim is None:
        dim = list(range(x.ndim))
    shift = [(x.shape[d] + 1) // 2 for d in dim]
    return roll(x, shift, dim)


def fft2c(data: torch.Tensor) -> torch.Tensor:
    _validate_complex_tensor(data)

    shifted = ifftshift(data, dim=[-3, -2])
    transformed = torch.view_as_real(
        torch.fft.fftn(torch.view_as_complex(shifted.contiguous()), dim=(-2, -1), norm="ortho")
    )
    return fftshift(transformed, dim=[-3, -2])


def ifft2c(data: torch.Tensor, *, need_shift: bool = True) -> torch.Tensor:
    _validate_complex_tensor(data)

    shifted = ifftshift(data, dim=[-3, -2])
    transformed = torch.view_as_real(
        torch.fft.ifftn(torch.view_as_complex(shifted.contiguous()), dim=(-2, -1), norm="ortho")
    )
    if need_shift:
        return fftshift(transformed, dim=[-3, -2])
    return transformed


# Compatibility aliases with the source code naming.
fft2c_new = fft2c
ifft2c_new = ifft2c
