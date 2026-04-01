from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, pe_dim: int = 128, magnify: float = 100.0) -> None:
        super().__init__()
        self.dim = pe_dim
        div_term = torch.exp(
            torch.arange(0, self.dim / 2, 2) * -(2 * math.log(10000.0) / self.dim)
        )
        self.register_buffer("div_term", div_term, persistent=False)
        self.magnify = magnify

    def forward(self, p_norm: torch.Tensor) -> torch.Tensor:
        p = p_norm * self.magnify

        no_batch = False
        if p.dim() == 2:
            no_batch = True
            p = p.unsqueeze(0)

        p_x = p[:, :, 0].unsqueeze(2)
        p_y = p[:, :, 1].unsqueeze(2)

        device = p.device
        dtype = p.dtype

        pe_x = torch.zeros(p_x.shape[0], p_x.shape[1], self.dim // 2, device=device, dtype=dtype)
        pe_y = torch.zeros(p_x.shape[0], p_x.shape[1], self.dim // 2, device=device, dtype=dtype)

        div = self.div_term.to(device=device, dtype=dtype)
        pe_x[:, :, 0::2] = torch.sin(p_x * div)
        pe_x[:, :, 1::2] = torch.cos(p_x * div)

        pe_y[:, :, 0::2] = torch.sin(p_y * div)
        pe_y[:, :, 1::2] = torch.cos(p_y * div)

        pe = torch.cat([pe_x, pe_y], dim=2)

        if no_batch:
            pe = pe.squeeze(0)
        return pe
