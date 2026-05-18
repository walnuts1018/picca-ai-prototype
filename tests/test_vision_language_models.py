from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from picca_search.infrastructure.vision_language_models import Florence2Captioner


def test_florence_captioner_uses_compatibility_options_for_remote_model(
    monkeypatch,
) -> None:
    captured_model_kwargs = {}
    captured_generate_kwargs = {}

    class FakeModel:
        def to(self, device):
            return self

        def eval(self) -> None:
            return None

        def generate(self, **kwargs):
            captured_generate_kwargs.update(kwargs)
            return [[1, 2, 3]]

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            captured_model_kwargs.update(kwargs)
            return FakeModel()

    class FakeProcessorFactory:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            return FakeProcessor()

    class FakeProcessor:
        def __call__(self, **kwargs):
            return FakeInputs(
                {
                    "input_ids": [1],
                    "pixel_values": [2],
                }
            )

        def batch_decode(self, generated_ids, skip_special_tokens):
            return ["caption"]

        def post_process_generation(self, generated_text, task, image_size):
            return {task: generated_text}

    class FakeInputs(dict):
        def to(self, device, dtype):
            return self

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        float16="float16",
        float32="float32",
        no_grad=lambda: FakeNoGrad(),
    )
    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=FakeModelFactory,
        AutoProcessor=FakeProcessorFactory,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    captioner = Florence2Captioner(device="cpu")
    monkeypatch.setattr(
        "picca_search.infrastructure.vision_language_models.Image.open",
        lambda image_path: FakeImage(),
    )
    captioner.caption(Path("image.jpg"))

    assert captured_model_kwargs["attn_implementation"] == "eager"
    assert captured_generate_kwargs["use_cache"] is False


class FakeImage:
    width = 640
    height = 480

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def convert(self, mode):
        return self


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None
