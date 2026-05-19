from __future__ import annotations

from fastapi import File, UploadFile

from picca_search.infrastructure.vision_language_models import PaddleOcrVlTextExtractor
from picca_search.services.common import TextResponse, create_service_app, upload_to_tempfile


def create_ocr_app(extractor: PaddleOcrVlTextExtractor) -> object:
    app = create_service_app("ocr-service")

    @app.post("/extract", response_model=TextResponse)
    async def extract(image: UploadFile = File(...)) -> TextResponse:
        temp_path = await upload_to_tempfile(image)
        try:
            return TextResponse(text=extractor.extract_text(temp_path))
        finally:
            temp_path.unlink(missing_ok=True)

    return app
