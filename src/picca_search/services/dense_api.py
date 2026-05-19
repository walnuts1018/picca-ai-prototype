from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import File, UploadFile

from picca_search.infrastructure.embedding_models import WaonSiglipEncoder
from picca_search.services.common import DenseVectorsResponse, TextBatchRequest, create_service_app


def create_dense_app(encoder: WaonSiglipEncoder) -> object:
    app = create_service_app("dense-service")

    @app.post("/encode/text-batch", response_model=DenseVectorsResponse)
    def encode_text_batch(request: TextBatchRequest) -> DenseVectorsResponse:
        return DenseVectorsResponse(vectors=[list(encoder.encode_text(text).values) for text in request.texts])

    @app.post("/encode/image-batch", response_model=DenseVectorsResponse)
    async def encode_image_batch(images: list[UploadFile] = File(...)) -> DenseVectorsResponse:
        temp_paths: list[Path] = []
        try:
            for image in images:
                suffix = Path(image.filename or "image.jpg").suffix or ".jpg"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
                    temp_path = Path(temporary_file.name)
                    temporary_file.write(await image.read())
                temp_paths.append(temp_path)
            vectors = encoder.encode_images_from_paths(temp_paths)
            return DenseVectorsResponse(vectors=[list(vector.values) for vector in vectors])
        finally:
            for path in temp_paths:
                path.unlink(missing_ok=True)

    return app
