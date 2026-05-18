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
PP_OCR_V5_MOBILE_DET_MODEL = "PP-OCRv5_mobile_det"
FLORENCE2_MODEL = "microsoft/Florence-2-base-ft"
FLORENCE2_MORE_DETAILED_CAPTION = "<MORE_DETAILED_CAPTION>"
CAT_TRANSLATE_MODEL = "cyberagent/CAT-Translate-0.8b"
CAT_TRANSLATE_PROMPT = "Translate the following {src_lang} text into {tgt_lang}.\n\n{src_text}"
CAT_SRC_LANG = "English"
CAT_TGT_LANG = "Japanese"


class PaddleOcrVlTextExtractor:
    def __init__(
        self,
        pipeline_version: str = PADDLE_OCR_VL_PIPELINE_VERSION,
        text_detector_model_name: str = PP_OCR_V5_MOBILE_DET_MODEL,
        text_detector_thresh: float = 0.15,
        text_detector_box_thresh: float = 0.25,
        text_detector_limit_side_len: int = 960,
        min_box_area_ratio: float = 0.00002,
        text_detector: Any | None = None,
        vl_pipeline: Any | None = None,
        **pipeline_options: Any,
    ) -> None:
        self.text_detector_thresh = text_detector_thresh
        self.text_detector_box_thresh = text_detector_box_thresh
        self.text_detector_limit_side_len = text_detector_limit_side_len
        self.min_box_area_ratio = min_box_area_ratio

        if text_detector is None or vl_pipeline is None:
            _prepare_paddlex_dependency_checks_for_ocr()
            PaddleOCRVL, TextDetection = import_module_symbols(
                "paddleocr",
                "PaddleOCRVL",
                "TextDetection",
                hidden_import_packages=OPTIONAL_PADDLEOCR_IMPORT_PROBES,
                hidden_distribution_names=OPTIONAL_PADDLEOCR_DISTRIBUTIONS,
            )
            if text_detector is None:
                text_detector = TextDetection(model_name=text_detector_model_name)
            if vl_pipeline is None:
                vl_pipeline = PaddleOCRVL(
                    pipeline_version=pipeline_version, **pipeline_options
                )

        self.text_detector = text_detector
        self.pipeline = vl_pipeline

    def extract_text(self, image_path: Path) -> str:
        if not self._should_run_vl(image_path):
            return ""
        output = self.pipeline.predict(str(image_path))
        return _text_from_paddleocr_result(output)

    def _should_run_vl(self, image_path: Path) -> bool:
        image_area = _image_area(image_path)
        results = self.text_detector.predict(
            input=str(image_path),
            batch_size=1,
            limit_side_len=self.text_detector_limit_side_len,
            limit_type="max",
            thresh=self.text_detector_thresh,
            box_thresh=self.text_detector_box_thresh,
        )
        for result in results:
            for box in _extract_detection_boxes(result):
                if _box_area_ratio(box, image_area) >= self.min_box_area_ratio:
                    return True
        return False


class Florence2Captioner:
    def __init__(
        self,
        model_name: str = FLORENCE2_MODEL,
        task_prompt: str = FLORENCE2_MORE_DETAILED_CAPTION,
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


class JapaneseTranslator:
    def __init__(
        self,
        model_name: str = CAT_TRANSLATE_MODEL,
        src_lang: str = CAT_SRC_LANG,
        tgt_lang: str = CAT_TGT_LANG,
        device: str | None = None,
        max_new_tokens: int = 128,
    ) -> None:
        import torch

        AutoModelForCausalLM, AutoTokenizer = import_transformers_symbols(
            "AutoModelForCausalLM",
            "AutoTokenizer",
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
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_new_tokens = max_new_tokens
        self.prompt_template = CAT_TRANSLATE_PROMPT
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
        ).to(self.device)
        self.model.eval()

    def translate(self, text: str) -> str:
        prompt = self.prompt_template.format(
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            src_text=text,
        )
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.device)
        with self.torch.no_grad():
            generated_ids = self.model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated_text = self.tokenizer.decode(
            generated_ids[0, inputs.shape[1]:],
            skip_special_tokens=True,
        )
        return generated_text.strip()


class Florence2WithJapaneseTranslation:
    def __init__(
        self,
        captioner: Florence2Captioner | None = None,
        translator: JapaneseTranslator | None = None,
        **kwargs: Any,
    ) -> None:
        self.captioner = captioner or Florence2Captioner(**kwargs)
        self.translator = translator or JapaneseTranslator(
            device=kwargs.get("device")
        )

    def caption(self, image_path: Path) -> str:
        english_caption = self.captioner.caption(image_path)
        if not english_caption:
            return ""
        return self.translator.translate(english_caption)


def _text_from_paddleocr_result(result: Any) -> str:
    values = list(_walk_text_values(result))
    return "\n".join(value for value in values if value != "")


def _caption_text_from_florence_answer(answer: Any) -> str:
    if isinstance(answer, str):
        return answer.strip()
    if isinstance(answer, dict):
        for key in (FLORENCE2_MORE_DETAILED_CAPTION, "<DETAILED_CAPTION>", "<CAPTION>"):
            value = answer.get(key)
            if isinstance(value, str) and value.strip() != "":
                return value.strip()
        for value in answer.values():
            if isinstance(value, str) and value.strip() != "":
                return value.strip()
    return ""


def _image_area(image_path: Path) -> int:
    with Image.open(image_path) as image:
        width, height = image.size
    return max(width * height, 1)


def _extract_detection_boxes(result: Any) -> list[Any]:
    data = getattr(result, "json", result)
    if callable(data):
        data = data()
    if not isinstance(data, dict):
        return []
    payload = data.get("res")
    if isinstance(payload, dict):
        data = payload
    return list(
        data.get("dt_polys")
        or data.get("boxes")
        or data.get("rec_boxes")
        or data.get("poly")
        or []
    )


def _box_area_ratio(box: Any, image_area: int) -> float:
    points = list(_iter_points(box))
    if len(points) == 0:
        return 0.0
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return max(width, 0.0) * max(height, 0.0) / image_area


def _iter_points(box: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if not isinstance(box, list | tuple):
        return points
    for point in box:
        if (
            isinstance(point, list | tuple)
            and len(point) >= 2
            and isinstance(point[0], int | float)
            and isinstance(point[1], int | float)
        ):
            points.append((float(point[0]), float(point[1])))
    return points


def _walk_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if isinstance(value, dict):
        texts: list[str] = []
        for key in (
            "markdown",
            "text",
            "rec_text",
            "transcription",
            "content",
            "block_content",
        ):
            texts.extend(_walk_text_values(value.get(key)))
        for key in ("res", "result", "data", "parsing_res_list"):
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
        "block_content",
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
