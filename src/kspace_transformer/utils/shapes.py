from __future__ import annotations

from kspace_transformer.data.types import BatchTensors, ForwardOutputs


def assert_batch_shapes(batch: BatchTensors) -> None:
    batch.validate()


def assert_model_output_shapes(outputs: ForwardOutputs) -> None:
    outputs.validate()
