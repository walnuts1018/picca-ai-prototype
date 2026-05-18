from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from PIL import Image

from picca_search.infrastructure.transformers_compat import (
    OPTIONAL_PADDLEOCR_DISTRIBUTIONS,
    OPTIONAL_PADDLEOCR_IMPORT_PROBES,
    import_module_symbols,
    import_transformers_symbols,
)


PADDLE_OCR_VL_PIPELINE_VERSION = "v1"
FLORENCE2_MODEL = "aipib/Florence-2-VQAJP2"
FLORENCE2_CAPTION = "<CAPTION>"


class PaddleOcrVlTextExtractor:
    def __init__(
        self,
        pipeline_version: str = PADDLE_OCR_VL_PIPELINE_VERSION,
        **pipeline_options: Any,
    ) -> None:
        (PaddleOCRVL,) = import_module_symbols(
            "paddleocr",
            "PaddleOCRVL",
            hidden_import_packages=OPTIONAL_PADDLEOCR_IMPORT_PROBES,
            hidden_distribution_names=OPTIONAL_PADDLEOCR_DISTRIBUTIONS,
        )
        _prepare_paddlex_dependency_checks_for_ocr()

        self.pipeline = PaddleOCRVL(
            pipeline_version=pipeline_version, **pipeline_options
        )

    def extract_text(self, image_path: Path) -> str:
        output = self.pipeline.predict(str(image_path))
        return _text_from_paddleocr_result(output)


class Florence2Captioner:
    def __init__(
        self,
        model_name: str = FLORENCE2_MODEL,
        task_prompt: str = FLORENCE2_CAPTION,
        device: str | None = None,
        max_new_tokens: int = 1024,
        num_beams: int = 3,
        attn_implementation: str = "eager",
        use_cache: bool = False,
    ) -> None:
        import torch

        AutoModelForCausalLM, AutoProcessor = import_transformers_symbols(
            "AutoModelForCausalLM",
            "AutoProcessor",
        )

        self.torch = torch
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.task_prompt = task_prompt
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.use_cache = use_cache
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        ).to(self.device)
        self.model.eval()

    def caption(self, image_path: Path) -> str:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            inputs = self.processor(
                text=self.task_prompt,
                images=rgb_image,
                return_tensors="pt",
            ).to(self.device, self.torch_dtype)
            image_size = (rgb_image.width, rgb_image.height)
        with self.torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                use_cache=self.use_cache,
            )
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=self.task_prompt,
            image_size=image_size,
        )
        return _caption_text_from_florence_answer(parsed_answer)


def _text_from_paddleocr_result(result: Any) -> str:
    values = list(_walk_text_values(result))
    return "\n".join(value for value in values if value != "")


def _caption_text_from_florence_answer(answer: Any) -> str:
    if isinstance(answer, str):
        return answer.strip()
    if isinstance(answer, dict):
        for key in (FLORENCE2_CAPTION, "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"):
            value = answer.get(key)
            if isinstance(value, str) and value.strip() != "":
                return value.strip()
        for value in answer.values():
            if isinstance(value, str) and value.strip() != "":
                return value.strip()
    return ""


def _walk_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if isinstance(value, dict):
        texts: list[str] = []
        for key in ("markdown", "text", "rec_text", "transcription", "content"):
            texts.extend(_walk_text_values(value.get(key)))
        for key in ("res", "result", "data"):
            texts.extend(_walk_text_values(value.get(key)))
        return texts
    if isinstance(value, list | tuple):
        texts = []
        for item in value:
            texts.extend(_walk_text_values(item))
        return texts

    texts = []
    for attribute in (
        "markdown",
        "text",
        "rec_text",
        "transcription",
        "content",
        "res",
    ):
        if hasattr(value, attribute):
            texts.extend(_walk_text_values(getattr(value, attribute)))
    return texts


def _prepare_paddlex_dependency_checks_for_ocr() -> None:
    deps = importlib.import_module("paddlex.utils.deps")
    original_is_dep_available = getattr(
        deps,
        "_picca_original_is_dep_available",
        deps.is_dep_available,
    )

    def is_dep_available(dep: str, /, check_version: bool = False) -> bool:
        if dep in OPTIONAL_PADDLEOCR_DISTRIBUTIONS:
            return True
        return original_is_dep_available(dep, check_version=check_version)

    deps._picca_original_is_dep_available = original_is_dep_available
    deps.is_dep_available = is_dep_available

    for cacheable in (original_is_dep_available, deps.is_extra_available):
        cacheable.cache_clear()
