from __future__ import annotations

import pytest

from picca_search.infrastructure.embedding_models import validate_sparse_onnx_output_names


def test_validate_sparse_onnx_output_names_accepts_logits() -> None:
    validate_sparse_onnx_output_names(["logits"])


def test_validate_sparse_onnx_output_names_rejects_missing_logits() -> None:
    with pytest.raises(ValueError, match="logits"):
        validate_sparse_onnx_output_names(["last_hidden_state"])
