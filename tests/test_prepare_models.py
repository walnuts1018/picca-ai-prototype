from __future__ import annotations

from scripts.prepare_models import SPLADE_ONNX_EXPORT_TASK


def test_splade_onnx_export_task_targets_masked_lm_logits() -> None:
    assert SPLADE_ONNX_EXPORT_TASK == "fill-mask"
