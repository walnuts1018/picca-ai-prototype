from __future__ import annotations

from fastapi import File, UploadFile

from picca_search.infrastructure.vision_language_models import Florence2WithJapaneseTranslation
from picca_search.services.common import TextResponse, create_service_app, upload_to_tempfile


def create_caption_app(captioner: Florence2WithJapaneseTranslation) -> object:
    app = create_service_app("caption-service")

    @app.post("/caption", response_model=TextResponse)
    async def caption(image: UploadFile = File(...)) -> TextResponse:
        temp_path = await upload_to_tempfile(image)
        try:
            return TextResponse(text=captioner.caption(temp_path))
        finally:
            temp_path.unlink(missing_ok=True)

    return app
