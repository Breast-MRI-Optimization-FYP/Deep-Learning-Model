from __future__ import annotations

import torch
from torch import nn

from kspace_transformer.utils.fftc import fft2c, ifft2c


def fill_in_k(
    masked_k: torch.Tensor,
    query_pos: torch.Tensor,
    query_result: torch.Tensor,
) -> torch.Tensor:
    if query_pos.ndim != 3 or query_pos.shape[-1] != 2:
        raise ValueError("query_pos must have shape [B,Q,2]")
    if query_result.ndim != 3 or query_result.shape[-1] != 2:
        raise ValueError("query_result must have shape [B,Q,2]")

    if query_pos.shape[:2] != query_result.shape[:2]:
        raise ValueError("query_pos and query_result must align on [B,Q]")

    out = masked_k.clone()
    h, w = out.shape[1], out.shape[2]

    rows = query_pos[:, :, 0].to(torch.int64).clamp(0, h - 1)
    cols = query_pos[:, :, 1].to(torch.int64).clamp(0, w - 1)

    for batch_idx in range(out.shape[0]):
        out[batch_idx, rows[batch_idx], cols[batch_idx], 0] = query_result[batch_idx, :, 0]
        out[batch_idx, rows[batch_idx], cols[batch_idx], 1] = query_result[batch_idx, :, 1]

    return out


def gather_k_values(k_data: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:
    if query_pos.ndim != 3 or query_pos.shape[-1] != 2:
        raise ValueError("query_pos must have shape [B,Q,2]")

    h, w = k_data.shape[1], k_data.shape[2]
    rows = query_pos[:, :, 0].to(torch.int64).clamp(0, h - 1)
    cols = query_pos[:, :, 1].to(torch.int64).clamp(0, w - 1)

    values = []
    for batch_idx in range(k_data.shape[0]):
        values.append(k_data[batch_idx, rows[batch_idx], cols[batch_idx], :])
    return torch.stack(values, dim=0)


def data_consistency(k_rec: torch.Tensor, k_sampled: torch.Tensor, mask_unsampled: torch.Tensor) -> torch.Tensor:
    k_rec_masked = k_rec * mask_unsampled
    return k_rec_masked + k_sampled


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        mid_channels: int = 48,
        num_convs: int = 4,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        convs: list[nn.Module] = []

        convs.append(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        )
        convs.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        for _ in range(num_convs):
            convs.append(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            )
            convs.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        convs.append(nn.Conv2d(mid_channels, in_channels, kernel_size=1))
        self.convs = nn.ModuleList(convs)

    def forward(
        self,
        k_sampled: torch.Tensor,
        i_in: torch.Tensor,
        mask_unsampled: torch.Tensor,
    ) -> torch.Tensor:
        output = i_in.permute(0, 3, 1, 2).contiguous()
        for layer in self.convs:
            output = layer(output)

        output = output.permute(0, 2, 3, 1).contiguous()
        output = output + i_in

        k_rec = fft2c(output)
        k_rec = data_consistency(k_rec, k_sampled, mask_unsampled)
        output = ifft2c(k_rec)
        return output
