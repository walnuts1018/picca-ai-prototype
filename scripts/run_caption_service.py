from __future__ import annotations

import os

import uvicorn

from picca_search.infrastructure.vision_language_models import Florence2WithJapaneseTranslation
from picca_search.services.caption_api import create_caption_app


def main() -> None:
    device = os.getenv("MODEL_DEVICE")
    port = int(os.getenv("PORT", "8004"))
    florence_model = os.getenv("FLORENCE2_MODEL_NAME", "microsoft/Florence-2-base-ft")
    translate_model = os.getenv("TRANSLATE_MODEL_NAME", "cyberagent/CAT-Translate-0.8b")
    captioner = Florence2WithJapaneseTranslation(
        model_name=florence_model,
        translator_model_name=translate_model,
        device=device
    )
    uvicorn.run(create_caption_app(captioner), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
