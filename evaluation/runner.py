from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from training.stage import TrainingStage
from utils.device import to_device
from utils.fftc import ifft2c
from utils.perf import RuntimeTracker

from .metrics import magnitude_images, per_sample_nmse, per_sample_psnr, per_sample_ssim
from .schemas import (
    EvaluationConfig,
    EvaluationResult,
    MaskMetadata,
    QualitativeSample,
    SampleMetricRecord,
)


class EvaluationRunner:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        device: torch.device,
        mask_metadata: list[MaskMetadata],
        config: EvaluationConfig,
    ) -> None:
        config.validate()
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.mask_metadata = {item.mask_id: item for item in mask_metadata}

    def _forward(self, batch: dict[str, Any], stage: TrainingStage) -> Any:
        return self.model(
            src=batch["sampled_k"],
            lr_pos=batch["LR_pos_norm"],
            src_pos=batch["sampled_pos_norm"],
            hr_pos=batch["unsampled_pos_norm"],
            k_us=batch["k_us"],
            unsampled_pos=batch["unsampled_pos"],
            up_scale=self.config.up_scale,
            mask=batch["selected_mask"],
            conv_weight=self.config.conv_weight,
            stage=stage.value,
        )

    def run(self, loader: Any) -> EvaluationResult:
        self.model.eval()
        records: list[SampleMetricRecord] = []
        qualitative: list[QualitativeSample] = []
        qualitative_counts: dict[str, int] = defaultdict(int)

        tracker = RuntimeTracker(device=self.device)
        tracker.start()

        with torch.no_grad():
            for batch in loader:
                batch = to_device(batch, self.device)
                predictions: dict[str, torch.Tensor] = {}
                if "K" in self.config.stages:
                    k_outputs = self._forward(batch, TrainingStage.K)
                    predictions["K"] = k_outputs.hr_transformer_images[-1]
                if "RM" in self.config.stages:
                    rm_outputs = self._forward(batch, TrainingStage.RM)
                    predictions["RM"] = rm_outputs.hr_refined_images[-1]

                target = batch["i_gt"]
                sample_indices = batch["sample_index"].detach().cpu().tolist()
                mask_ids = batch["mask_id"].detach().cpu().tolist()
                actual_accelerations = batch["actual_acceleration"].detach().cpu().tolist()

                for stage, prediction in predictions.items():
                    psnr_values = per_sample_psnr(prediction, target)
                    ssim_values = per_sample_ssim(prediction, target)
                    nmse_values = per_sample_nmse(prediction, target)

                    for idx, (sample_index, mask_id) in enumerate(zip(sample_indices, mask_ids)):
                        metadata = self.mask_metadata[int(mask_id)]
                        records.append(
                            SampleMetricRecord(
                                sample_index=int(sample_index),
                                mask_id=int(mask_id),
                                acceleration=float(metadata.acceleration),
                                acceleration_label=metadata.acceleration_label,
                                actual_acceleration=float(actual_accelerations[idx]),
                                sampling_pattern=metadata.sampling_pattern,
                                stage=stage,
                                psnr=float(psnr_values[idx]),
                                ssim=float(ssim_values[idx]),
                                nmse=float(nmse_values[idx]),
                            )
                        )

                if self.config.qualitative_samples_per_acceleration <= 0:
                    continue

                undersampled_mag = magnitude_images(ifft2c(batch["k_us"], need_shift=True))
                target_mag = magnitude_images(target)
                prediction_magnitudes = {
                    stage: magnitude_images(prediction) for stage, prediction in predictions.items()
                }
                for idx, (sample_index, mask_id) in enumerate(zip(sample_indices, mask_ids)):
                    metadata = self.mask_metadata[int(mask_id)]
                    label = metadata.acceleration_label
                    if (
                        qualitative_counts[label]
                        >= self.config.qualitative_samples_per_acceleration
                    ):
                        continue
                    qualitative.append(
                        QualitativeSample(
                            sample_index=int(sample_index),
                            mask_id=int(mask_id),
                            acceleration_label=label,
                            undersampled=undersampled_mag[idx],
                            ground_truth=target_mag[idx],
                            predictions={
                                stage: images[idx]
                                for stage, images in prediction_magnitudes.items()
                            },
                        )
                    )
                    qualitative_counts[label] += 1

        return EvaluationResult(
            records=records,
            qualitative_samples=qualitative,
            runtime_seconds=tracker.elapsed_seconds(),
            peak_memory_bytes=tracker.peak_memory_bytes(),
        )
