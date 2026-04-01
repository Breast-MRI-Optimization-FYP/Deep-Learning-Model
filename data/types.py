from __future__ import annotations

from dataclasses import dataclass, field

import torch


def _assert_rank(name: str, tensor: torch.Tensor, expected_rank: int) -> None:
    if tensor.ndim != expected_rank:
        raise ValueError(
            f"{name} must have rank {expected_rank}, got shape {tuple(tensor.shape)}"
        )


def _assert_last_dim(name: str, tensor: torch.Tensor, expected: int = 2) -> None:
    if tensor.shape[-1] != expected:
        raise ValueError(f"{name} last dim must be {expected}, got shape {tuple(tensor.shape)}")


def _assert_integer_dtype(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype not in {torch.int32, torch.int64}:
        raise ValueError(f"{name} must use integer dtype, got {tensor.dtype}")


def _assert_float_or_bool(name: str, tensor: torch.Tensor) -> None:
    if not (torch.is_floating_point(tensor) or tensor.dtype == torch.bool):
        raise ValueError(f"{name} must be float or bool, got {tensor.dtype}")


@dataclass(slots=True)
class TokenizedSample:
    sampled_k: torch.Tensor
    sampled_pos: torch.Tensor
    sampled_pos_norm: torch.Tensor
    unsampled_pos: torch.Tensor
    unsampled_pos_norm: torch.Tensor
    k_us: torch.Tensor
    selected_mask: torch.Tensor
    k_gt: torch.Tensor | None = None
    i_gt: torch.Tensor | None = None
    lr_i_gt: torch.Tensor | None = None
    lr_k_gt: torch.Tensor | None = None
    lr_pos: torch.Tensor | None = None
    lr_pos_norm: torch.Tensor | None = None

    def validate(self) -> None:
        _assert_rank("sampled_k", self.sampled_k, 2)
        _assert_last_dim("sampled_k", self.sampled_k)

        _assert_rank("sampled_pos", self.sampled_pos, 2)
        _assert_last_dim("sampled_pos", self.sampled_pos)
        _assert_integer_dtype("sampled_pos", self.sampled_pos)

        _assert_rank("sampled_pos_norm", self.sampled_pos_norm, 2)
        _assert_last_dim("sampled_pos_norm", self.sampled_pos_norm)

        _assert_rank("unsampled_pos", self.unsampled_pos, 2)
        _assert_last_dim("unsampled_pos", self.unsampled_pos)
        _assert_integer_dtype("unsampled_pos", self.unsampled_pos)

        _assert_rank("unsampled_pos_norm", self.unsampled_pos_norm, 2)
        _assert_last_dim("unsampled_pos_norm", self.unsampled_pos_norm)

        _assert_rank("k_us", self.k_us, 3)
        _assert_last_dim("k_us", self.k_us)

        _assert_rank("selected_mask", self.selected_mask, 3)
        _assert_last_dim("selected_mask", self.selected_mask)
        _assert_float_or_bool("selected_mask", self.selected_mask)

        if self.k_gt is not None:
            _assert_rank("k_gt", self.k_gt, 3)
            _assert_last_dim("k_gt", self.k_gt)
        if self.i_gt is not None:
            _assert_rank("i_gt", self.i_gt, 3)
            _assert_last_dim("i_gt", self.i_gt)
        if self.lr_i_gt is not None:
            _assert_rank("lr_i_gt", self.lr_i_gt, 3)
            _assert_last_dim("lr_i_gt", self.lr_i_gt)
        if self.lr_k_gt is not None:
            _assert_rank("lr_k_gt", self.lr_k_gt, 3)
            _assert_last_dim("lr_k_gt", self.lr_k_gt)
        if self.lr_pos is not None:
            _assert_rank("lr_pos", self.lr_pos, 2)
            _assert_last_dim("lr_pos", self.lr_pos)
            _assert_integer_dtype("lr_pos", self.lr_pos)
        if self.lr_pos_norm is not None:
            _assert_rank("lr_pos_norm", self.lr_pos_norm, 2)
            _assert_last_dim("lr_pos_norm", self.lr_pos_norm)


@dataclass(slots=True)
class BatchTensors:
    sampled_k: torch.Tensor
    sampled_pos_norm: torch.Tensor
    unsampled_pos: torch.Tensor
    unsampled_pos_norm: torch.Tensor
    k_us: torch.Tensor
    i_gt: torch.Tensor
    k_gt: torch.Tensor
    selected_mask: torch.Tensor
    lr_i_gt: torch.Tensor
    lr_k_gt: torch.Tensor
    lr_pos_norm: torch.Tensor
    sampled_pos: torch.Tensor | None = None
    lr_pos: torch.Tensor | None = None

    def validate(self) -> None:
        _assert_rank("sampled_k", self.sampled_k, 3)
        _assert_last_dim("sampled_k", self.sampled_k)

        _assert_rank("sampled_pos_norm", self.sampled_pos_norm, 3)
        _assert_last_dim("sampled_pos_norm", self.sampled_pos_norm)

        _assert_rank("unsampled_pos", self.unsampled_pos, 3)
        _assert_last_dim("unsampled_pos", self.unsampled_pos)
        _assert_integer_dtype("unsampled_pos", self.unsampled_pos)

        _assert_rank("unsampled_pos_norm", self.unsampled_pos_norm, 3)
        _assert_last_dim("unsampled_pos_norm", self.unsampled_pos_norm)

        for name, tensor in {
            "k_us": self.k_us,
            "i_gt": self.i_gt,
            "k_gt": self.k_gt,
            "selected_mask": self.selected_mask,
            "lr_i_gt": self.lr_i_gt,
            "lr_k_gt": self.lr_k_gt,
        }.items():
            _assert_rank(name, tensor, 4)
            _assert_last_dim(name, tensor)

        _assert_float_or_bool("selected_mask", self.selected_mask)

        _assert_rank("lr_pos_norm", self.lr_pos_norm, 3)
        _assert_last_dim("lr_pos_norm", self.lr_pos_norm)

        if self.sampled_pos is not None:
            _assert_rank("sampled_pos", self.sampled_pos, 3)
            _assert_last_dim("sampled_pos", self.sampled_pos)
            _assert_integer_dtype("sampled_pos", self.sampled_pos)

        if self.lr_pos is not None:
            _assert_rank("lr_pos", self.lr_pos, 3)
            _assert_last_dim("lr_pos", self.lr_pos)
            _assert_integer_dtype("lr_pos", self.lr_pos)

        batch_size = self.sampled_k.shape[0]
        for name, tensor in {
            "sampled_pos_norm": self.sampled_pos_norm,
            "unsampled_pos": self.unsampled_pos,
            "unsampled_pos_norm": self.unsampled_pos_norm,
            "k_us": self.k_us,
            "i_gt": self.i_gt,
            "k_gt": self.k_gt,
            "selected_mask": self.selected_mask,
            "lr_i_gt": self.lr_i_gt,
            "lr_k_gt": self.lr_k_gt,
            "lr_pos_norm": self.lr_pos_norm,
        }.items():
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"Batch dimension mismatch for {name}: expected {batch_size}, got {tensor.shape[0]}"
                )


@dataclass(slots=True)
class ForwardOutputs:
    lr_images: list[torch.Tensor] = field(default_factory=list)
    up_lr_image: torch.Tensor | None = None
    up_lr_k: torch.Tensor | None = None
    hr_transformer_images: list[torch.Tensor] = field(default_factory=list)
    hr_refined_images: list[torch.Tensor] = field(default_factory=list)

    def validate(self) -> None:
        batch_size: int | None = None

        for idx, tensor in enumerate(self.lr_images):
            _assert_rank(f"lr_images[{idx}]", tensor, 4)
            _assert_last_dim(f"lr_images[{idx}]", tensor)
            batch_size = tensor.shape[0] if batch_size is None else batch_size
            if tensor.shape[0] != batch_size:
                raise ValueError("lr_images batch size mismatch")

        if self.up_lr_image is not None:
            _assert_rank("up_lr_image", self.up_lr_image, 4)
            _assert_last_dim("up_lr_image", self.up_lr_image)
            batch_size = self.up_lr_image.shape[0] if batch_size is None else batch_size
            if self.up_lr_image.shape[0] != batch_size:
                raise ValueError("up_lr_image batch size mismatch")

        if self.up_lr_k is not None:
            _assert_rank("up_lr_k", self.up_lr_k, 4)
            _assert_last_dim("up_lr_k", self.up_lr_k)
            batch_size = self.up_lr_k.shape[0] if batch_size is None else batch_size
            if self.up_lr_k.shape[0] != batch_size:
                raise ValueError("up_lr_k batch size mismatch")

        for idx, tensor in enumerate(self.hr_transformer_images):
            _assert_rank(f"hr_transformer_images[{idx}]", tensor, 4)
            _assert_last_dim(f"hr_transformer_images[{idx}]", tensor)
            batch_size = tensor.shape[0] if batch_size is None else batch_size
            if tensor.shape[0] != batch_size:
                raise ValueError("hr_transformer_images batch size mismatch")

        for idx, tensor in enumerate(self.hr_refined_images):
            _assert_rank(f"hr_refined_images[{idx}]", tensor, 4)
            _assert_last_dim(f"hr_refined_images[{idx}]", tensor)
            batch_size = tensor.shape[0] if batch_size is None else batch_size
            if tensor.shape[0] != batch_size:
                raise ValueError("hr_refined_images batch size mismatch")


@dataclass(slots=True)
class LossBreakdown:
    total_loss: torch.Tensor
    lr_img_losses: list[float] = field(default_factory=list)
    lr_k_losses: list[float] = field(default_factory=list)
    hr_img_losses: list[float] = field(default_factory=list)
    hr_k_losses: list[float] = field(default_factory=list)
    rm_img_losses: list[float] = field(default_factory=list)
    rm_k_losses: list[float] = field(default_factory=list)

    def validate(self, *, expected_lr_layers: int | None = None, expected_hr_layers: int | None = None) -> None:
        _assert_rank("total_loss", self.total_loss, 0)

        def _check_finite(name: str, values: list[float]) -> None:
            for idx, value in enumerate(values):
                if not torch.isfinite(torch.tensor(value)):
                    raise ValueError(f"{name}[{idx}] must be finite, got {value!r}")

        _check_finite("lr_img_losses", self.lr_img_losses)
        _check_finite("lr_k_losses", self.lr_k_losses)
        _check_finite("hr_img_losses", self.hr_img_losses)
        _check_finite("hr_k_losses", self.hr_k_losses)
        _check_finite("rm_img_losses", self.rm_img_losses)
        _check_finite("rm_k_losses", self.rm_k_losses)

        if expected_lr_layers is not None:
            if len(self.lr_img_losses) != expected_lr_layers:
                raise ValueError("lr_img_losses length does not match expected_lr_layers")
            if len(self.lr_k_losses) != expected_lr_layers:
                raise ValueError("lr_k_losses length does not match expected_lr_layers")

        if expected_hr_layers is not None:
            for name, values in {
                "hr_img_losses": self.hr_img_losses,
                "hr_k_losses": self.hr_k_losses,
                "rm_img_losses": self.rm_img_losses,
                "rm_k_losses": self.rm_k_losses,
            }.items():
                if len(values) != expected_hr_layers:
                    raise ValueError(f"{name} length does not match expected_hr_layers")


@dataclass(slots=True)
class SplitIndices:
    train: torch.Tensor
    valid: torch.Tensor
    test: torch.Tensor

    def validate(self) -> None:
        for name, tensor in {"train": self.train, "valid": self.valid, "test": self.test}.items():
            _assert_rank(name, tensor, 1)
            _assert_integer_dtype(name, tensor)


@dataclass(slots=True)
class DatasetCache:
    sampled_k: list[torch.Tensor] = field(default_factory=list)
    sampled_pos: list[torch.Tensor] = field(default_factory=list)
    sampled_pos_norm: list[torch.Tensor] = field(default_factory=list)
    unsampled_pos: list[torch.Tensor] = field(default_factory=list)
    unsampled_pos_norm: list[torch.Tensor] = field(default_factory=list)
