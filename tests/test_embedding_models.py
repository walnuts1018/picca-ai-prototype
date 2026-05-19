from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from picca_search.infrastructure.embedding_models import SpladeJapaneseSparseEncoder


class _StopAfterTokenization(Exception):
    pass


class _FakeInputs(dict):
    def to(self, device: str):
        raise _StopAfterTokenization(device)


class _RecordingTokenizer:
    def __init__(self, *, model_max_length: int, unk_token_id: int = 99) -> None:
        self.model_max_length = model_max_length
        self.unk_token_id = unk_token_id
        self.calls: list[dict[str, object]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return _FakeInputs()


class _FakeMaskedLmModel:
    def __init__(self, *, max_position_embeddings: int) -> None:
        self.config = SimpleNamespace(max_position_embeddings=max_position_embeddings)

    def to(self, device: str):
        return self

    def eval(self) -> None:
        return None


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = ModuleType("torch")
    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def _install_fake_transformers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tokenizer: _RecordingTokenizer,
    max_position_embeddings: int,
) -> None:
    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str) -> _RecordingTokenizer:
            return tokenizer

    class _AutoModelForMaskedLM:
        @staticmethod
        def from_pretrained(model_name: str) -> _FakeMaskedLmModel:
            return _FakeMaskedLmModel(max_position_embeddings=max_position_embeddings)

    monkeypatch.setattr(
        "picca_search.infrastructure.embedding_models.import_transformers_symbols",
        lambda *names: (_AutoModelForMaskedLM, _AutoTokenizer),
    )


def test_splade_encode_text_truncates_to_model_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch)
    tokenizer = _RecordingTokenizer(model_max_length=8192)
    _install_fake_transformers(
        monkeypatch,
        tokenizer=tokenizer,
        max_position_embeddings=2048,
    )
    encoder = SpladeJapaneseSparseEncoder(device="cpu")

    with pytest.raises(_StopAfterTokenization):
        encoder.encode_text("x" * 10000)

    assert tokenizer.calls[0]["truncation"] is True
    assert tokenizer.calls[0]["max_length"] == 2048


def test_splade_encode_texts_truncates_to_model_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch)
    tokenizer = _RecordingTokenizer(model_max_length=8192)
    _install_fake_transformers(
        monkeypatch,
        tokenizer=tokenizer,
        max_position_embeddings=2048,
    )
    encoder = SpladeJapaneseSparseEncoder(device="cpu")

    with pytest.raises(_StopAfterTokenization):
        encoder.encode_texts(["a" * 10000, "b" * 10000])

    assert tokenizer.calls[0]["padding"] is True
    assert tokenizer.calls[0]["truncation"] is True
    assert tokenizer.calls[0]["max_length"] == 2048
