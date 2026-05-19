from __future__ import annotations

import os

import uvicorn

from picca_search.infrastructure.vision_language_models import PaddleOcrVlTextExtractor
from picca_search.services.ocr_api import create_ocr_app


def main() -> None:
    port = int(os.getenv("PORT", "8003"))
    extractor = PaddleOcrVlTextExtractor(device=os.getenv("MODEL_DEVICE"))
    uvicorn.run(create_ocr_app(extractor), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
