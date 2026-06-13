from __future__ import annotations

from typing import Any

import torch

from training.metrics import MetricsAccumulator, compute_nmse, compute_psnr, compute_ssim
from training.stage import TrainingStage
from utils.device import to_device


class InferenceRunner:
    def __init__(self, *, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.model.to(self.device)

    @staticmethod
    def _select_prediction(outputs: Any, stage: TrainingStage) -> torch.Tensor:
        if hasattr(outputs, "up_lr_image") and hasattr(outputs, "hr_transformer_images"):
            if stage is TrainingStage.LR:
                if outputs.up_lr_image is not None:
                    return outputs.up_lr_image
                return outputs.lr_images[-1]
            if stage is TrainingStage.K:
                return outputs.hr_transformer_images[-1]
            return outputs.hr_refined_images[-1]

        if isinstance(outputs, tuple) and len(outputs) == 5:
            lr_images, up_lr_image, _, hr_transformer, hr_refined = outputs
            if stage is TrainingStage.LR:
                return up_lr_image if up_lr_image is not None else lr_images[-1]
            if stage is TrainingStage.K:
                return hr_transformer[-1]
            return hr_refined[-1]

        raise TypeError("Unsupported model output type for inference")

    def run_test_set(
        self,
        loader: Any,
        *,
        stage: TrainingStage = TrainingStage.RM,
        up_scale: int = 2,
        conv_weight: float = 1.0,
    ) -> dict[str, float | int | str]:
        self.model.eval()
        acc = MetricsAccumulator()

        with torch.no_grad():
            for batch in loader:
                batch = to_device(batch, self.device)
                outputs = self.model(
                    src=batch["sampled_k"],
                    lr_pos=batch["LR_pos_norm"],
                    src_pos=batch["sampled_pos_norm"],
                    hr_pos=batch["unsampled_pos_norm"],
                    k_us=batch["k_us"],
                    unsampled_pos=batch["unsampled_pos"],
                    up_scale=up_scale,
                    mask=batch["selected_mask"],
                    conv_weight=conv_weight,
                    stage=stage.value,
                )

                prediction = self._select_prediction(outputs, stage)
                target = batch["i_gt"]
                batch_size = int(target.shape[0])

                psnr_sum = compute_psnr(prediction.detach(), target.detach())
                ssim_sum = compute_ssim(prediction.detach(), target.detach())
                nmse_sum = compute_nmse(prediction.detach(), target.detach())
                acc.update("psnr", float(psnr_sum), n=batch_size)
                acc.update("ssim", float(ssim_sum), n=batch_size)
                acc.update("nmse", float(nmse_sum), n=batch_size)

        return {
            "stage": stage.value,
            "mean_psnr": acc.mean("psnr"),
            "mean_ssim": acc.mean("ssim"),
            "mean_nmse": acc.mean("nmse"),
            "num_samples": acc.counts.get("psnr", 0),
        }

