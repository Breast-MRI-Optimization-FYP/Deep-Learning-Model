from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from kspace_transformer.data.types import ForwardOutputs
from kspace_transformer.utils.fftc import fft2c, ifft2c

from .blocks import CNNBlock, fill_in_k, gather_k_values
from .layers import (
    TransformerDecoderLayerHR,
    TransformerDecoderLayerLR,
    TransformerEncoderLayer,
)
from .positional_encoding import PositionalEncoding


def _get_clones(module: nn.Module, num_layers: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward(self, src: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        output = self.with_pos_embed(src, pos)
        for layer in self.layers:
            output = layer(output)
        return output


class TransformerDecoderLR(nn.Module):
    def __init__(
        self,
        d_model: int,
        lr_size: int,
        channel: int,
        decoder_layer: nn.Module,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.channel = channel
        self.lr_size = lr_size

        self.lr_predict_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(inplace=True),
                    nn.Linear(d_model, channel),
                )
                for _ in range(num_layers)
            ]
        )
        self.lr_norm_layers = nn.ModuleList(
            [nn.LayerNorm(d_model, eps=1e-6) for _ in range(num_layers)]
        )

    def forward(
        self,
        encoder_memory: torch.Tensor,
        lr_pe: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        transformer_interpredict: list[torch.Tensor] = []

        current = lr_pe
        for layer_idx, layer in enumerate(self.layers):
            output_memory = layer(current, encoder_memory)

            output = self.lr_predict_layers[layer_idx](
                self.lr_norm_layers[layer_idx](output_memory)
            )
            output = torch.reshape(
                output,
                (output.shape[0], self.lr_size, self.lr_size, self.channel),
            )
            output = ifft2c(output, need_shift=True)
            transformer_interpredict.append(output)

            current = output_memory

        return transformer_interpredict, output_memory


class TransformerDecoderHR(nn.Module):
    def __init__(
        self,
        d_model: int,
        channel: int,
        decoder_layer: nn.Module,
        num_layers: int,
        conv_channel: int = 64,
        conv_num: int = 3,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

        self.hr_predict_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(inplace=True),
                    nn.Linear(d_model, channel),
                )
                for _ in range(num_layers)
            ]
        )
        self.hr_norm_layers = nn.ModuleList(
            [nn.LayerNorm(d_model, eps=1e-6) for _ in range(num_layers)]
        )
        self.hr_embed_layer = nn.Sequential(
            nn.Linear(channel, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.conv_blocks = nn.ModuleList(
            [
                CNNBlock(
                    in_channels=channel,
                    mid_channels=conv_channel,
                    num_convs=conv_num,
                    kernel_size=kernel_size,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        *,
        lr_memory: torch.Tensor,
        query_pe: torch.Tensor,
        query_value: torch.Tensor,
        unsampled_pos: torch.Tensor,
        k_us: torch.Tensor,
        mask: torch.Tensor,
        conv_weight: float,
        stage: str,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        current = query_pe + self.hr_embed_layer(query_value)

        transformer_interpredict: list[torch.Tensor] = []
        cnn_interpredict: list[torch.Tensor] = []

        for layer_idx, layer in enumerate(self.layers):
            output_memory = layer(current, lr_memory)

            output = self.hr_predict_layers[layer_idx](
                self.hr_norm_layers[layer_idx](output_memory)
            )
            output = fill_in_k(k_us, unsampled_pos, output)
            output = ifft2c(output, need_shift=True)
            transformer_interpredict.append(output)

            if stage == "RM":
                conv_output = self.conv_blocks[layer_idx](k_us, output, mask)
                cnn_interpredict.append(conv_output)
            else:
                conv_output = torch.zeros_like(output)
                cnn_interpredict.append(conv_output)

            if layer_idx + 1 == len(self.layers):
                return transformer_interpredict, cnn_interpredict

            if stage == "RM":
                conv_k = fft2c(conv_output)
                unsampled_value = gather_k_values(conv_k, unsampled_pos)
                current = conv_weight * self.hr_embed_layer(unsampled_value) + output_memory
            else:
                current = output_memory

        return transformer_interpredict, cnn_interpredict


class KSpaceTransformer(nn.Module):
    def __init__(
        self,
        lr_size: int,
        channel: int = 2,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_lrdecoder_layers: int = 6,
        num_hrdecoder_layers: int = 6,
        dim_feedforward: int = 2048,
        hr_conv_channel: int = 64,
        hr_conv_num: int = 3,
        hr_kernel_size: int = 5,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.num_hrdecoder_layers = num_hrdecoder_layers

        self.encoder_embed_layer = nn.Sequential(
            nn.Linear(channel, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        self.pe_layer = PositionalEncoding(d_model, magnify=250.0)

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer_lr = TransformerDecoderLayerLR(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        self.decoder_lr = TransformerDecoderLR(
            d_model=d_model,
            lr_size=lr_size,
            channel=2,
            decoder_layer=decoder_layer_lr,
            num_layers=num_lrdecoder_layers,
        )

        decoder_layer_hr = TransformerDecoderLayerHR(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        self.decoder_hr = TransformerDecoderHR(
            d_model=d_model,
            channel=2,
            decoder_layer=decoder_layer_hr,
            num_layers=num_hrdecoder_layers,
            conv_channel=hr_conv_channel,
            conv_num=hr_conv_num,
            kernel_size=hr_kernel_size,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(
        self,
        src: torch.Tensor,
        lr_pos: torch.Tensor,
        src_pos: torch.Tensor,
        hr_pos: torch.Tensor,
        k_us: torch.Tensor,
        unsampled_pos: torch.Tensor,
        up_scale: int,
        mask: torch.Tensor,
        conv_weight: float,
        stage: str,
    ) -> ForwardOutputs:
        src_embed = self.encoder_embed_layer(src)
        src_pe = self.pe_layer(src_pos)
        encoder_memory = self.encoder(src_embed, pos=src_pe)

        lr_pe = self.pe_layer(lr_pos)
        lr_transformer_outputs, lr_memory = self.decoder_lr(encoder_memory, lr_pe)

        lr_image = lr_transformer_outputs[-1].permute(0, 3, 1, 2).contiguous()
        up_lr_i = (
            F.interpolate(lr_image, scale_factor=up_scale, mode="bicubic")
            .permute(0, 2, 3, 1)
            .contiguous()
        )
        up_lr_k = fft2c(up_lr_i)

        if stage == "LR":
            zeros = [torch.zeros_like(up_lr_i) for _ in range(self.num_hrdecoder_layers)]
            return ForwardOutputs(
                lr_images=lr_transformer_outputs,
                up_lr_image=up_lr_i,
                up_lr_k=up_lr_k,
                hr_transformer_images=zeros,
                hr_refined_images=zeros,
            )

        unsampled_value = gather_k_values(up_lr_k, unsampled_pos)
        hr_pe = self.pe_layer(hr_pos)
        hr_transformer_outputs, hr_conv_outputs = self.decoder_hr(
            lr_memory=lr_memory,
            query_pe=hr_pe,
            query_value=unsampled_value,
            unsampled_pos=unsampled_pos,
            k_us=k_us,
            mask=mask,
            conv_weight=conv_weight,
            stage=stage,
        )

        return ForwardOutputs(
            lr_images=lr_transformer_outputs,
            up_lr_image=up_lr_i,
            up_lr_k=up_lr_k,
            hr_transformer_images=hr_transformer_outputs,
            hr_refined_images=hr_conv_outputs,
        )
