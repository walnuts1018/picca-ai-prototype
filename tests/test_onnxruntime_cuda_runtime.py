from __future__ import annotations

import sys

import pytest

from picca_search.infrastructure.transformers_compat import ort_provider_for_device


class _FakeOrt:
    def __init__(self, providers: list[str]) -> None:
        self._providers = providers

    def get_available_providers(self) -> list[str]:
        return self._providers


def test_ort_provider_for_device_returns_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        _FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )

    assert ort_provider_for_device("cuda", require_accelerator=True) == "CUDAExecutionProvider"


def test_ort_provider_for_device_raises_when_cuda_required_but_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrt(["CPUExecutionProvider"]))

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        ort_provider_for_device("cuda", require_accelerator=True)


def test_model_cuda_dockerfile_installs_onnxruntime_gpu() -> None:
    dockerfile = open("model-cuda.Dockerfile", encoding="utf-8").read()

    assert "onnxruntime-gpu" in dockerfile


def test_model_cuda_dockerfile_uses_cuda_12_base_image() -> None:
    dockerfile = open("model-cuda.Dockerfile", encoding="utf-8").read()

    assert "FROM nvidia/cuda:12." in dockerfile
