from __future__ import annotations

from dataclasses import dataclass

from .types import TokenizedSample


@dataclass(slots=True)
class TruncationStats:
    total_samples: int = 0
    truncated_samples: int = 0
    max_observed_len: int = 0
    sampled_tokens_before: int = 0
    sampled_tokens_after: int = 0


class SequenceLimiter:
    def __init__(self, max_seq_len: int) -> None:
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer")
        self.max_seq_len = max_seq_len

    def truncate(
        self,
        sample: TokenizedSample,
        max_seq_len: int | None = None,
        *,
        stats: TruncationStats | None = None,
    ) -> TokenizedSample:
        limit = self.max_seq_len if max_seq_len is None else max_seq_len
        if limit <= 0:
            raise ValueError("max_seq_len must be a positive integer")

        sampled_before = sample.sampled_k.shape[0]
        unsampled_before = sample.unsampled_pos.shape[0]

        truncated = TokenizedSample(
            sampled_k=sample.sampled_k[:limit],
            sampled_pos=sample.sampled_pos[:limit],
            sampled_pos_norm=sample.sampled_pos_norm[:limit],
            unsampled_pos=sample.unsampled_pos[:limit],
            unsampled_pos_norm=sample.unsampled_pos_norm[:limit],
            k_us=sample.k_us,
            selected_mask=sample.selected_mask,
            k_gt=sample.k_gt,
            i_gt=sample.i_gt,
            lr_i_gt=sample.lr_i_gt,
            lr_k_gt=sample.lr_k_gt,
            lr_pos=sample.lr_pos,
            lr_pos_norm=sample.lr_pos_norm,
        )

        if stats is not None:
            stats.total_samples += 1
            stats.max_observed_len = max(stats.max_observed_len, sampled_before, unsampled_before)
            stats.sampled_tokens_before += sampled_before
            stats.sampled_tokens_after += int(truncated.sampled_k.shape[0])
            if sampled_before > limit or unsampled_before > limit:
                stats.truncated_samples += 1

        truncated.validate()
        return truncated


def report_truncation_stats(stats: TruncationStats) -> dict[str, float | int]:
    ratio = 0.0
    if stats.total_samples > 0:
        ratio = stats.truncated_samples / stats.total_samples

    sampled_retention = 1.0
    if stats.sampled_tokens_before > 0:
        sampled_retention = stats.sampled_tokens_after / stats.sampled_tokens_before

    return {
        "total_samples": stats.total_samples,
        "truncated_samples": stats.truncated_samples,
        "truncation_ratio": ratio,
        "max_observed_len": stats.max_observed_len,
        "sampled_retention_ratio": sampled_retention,
    }
