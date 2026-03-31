from __future__ import annotations

import torch.nn.functional as F
from torch import nn

from .attention import MultiHeadAttention


def get_activation(name: str):
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {name}")


def _get_activation_module(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "glu":
        return nn.GLU()
    raise RuntimeError(f"activation should be relu/gelu/glu, not {name}")


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(
            nhead,
            d_model,
            d_k=d_model // nhead,
            d_v=d_model // nhead,
            dropout=dropout,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.activation = get_activation(activation)

    def forward(self, src):
        q = k = src
        src = self.self_attn(q, k, src)[0]

        src2 = self.norm(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayerLR(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttention(
            nhead,
            d_model,
            d_k=d_model // nhead,
            d_v=d_model // nhead,
            dropout=dropout,
        )
        self.multihead_attn = MultiHeadAttention(
            nhead,
            d_model,
            d_k=d_model // nhead,
            d_v=d_model // nhead,
            dropout=dropout,
        )

        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            _get_activation_module(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory):
        tgt = self.multihead_attn(tgt, memory, memory)[0]
        tgt = self.self_attn(tgt, tgt, tgt)[0]

        tgt2 = self.norm1(tgt)
        tgt2 = self.ffn1(tgt2)
        tgt = tgt + tgt2
        return tgt


class TransformerDecoderLayerHR(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.multihead_attn = MultiHeadAttention(
            nhead,
            d_model,
            d_k=d_model // nhead,
            d_v=d_model // nhead,
            dropout=dropout,
        )

        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            _get_activation_module(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory):
        tgt = self.multihead_attn(tgt, memory, memory)[0]

        tgt2 = self.norm1(tgt)
        tgt2 = self.ffn1(tgt2)
        tgt = tgt + tgt2
        return tgt
