from __future__ import annotations

import argparse
from pathlib import Path

from qdrant_client import QdrantClient

from picca_search.application import ingest_image_with_extracted_text
from picca_search.domain import SUPPORTED_IMAGE_EXTENSIONS
from picca_search.infrastructure.embedding_models import (
    SpladeJapaneseSparseEncoder,
    WaonSiglipEncoder,
)
from picca_search.infrastructure.qdrant_index import QdrantImageIndex
from picca_search.infrastructure.vision_language_models import (
    Florence2Captioner,
    PaddleOcrVlTextExtractor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local images into Qdrant.")
    parser.add_argument("image_dir", type=Path, help="Directory containing image files.")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="picca_images")
    args = parser.parse_args()

    images = sorted(
        path
        for path in args.image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if len(images) == 0:
        raise SystemExit(f"No supported images found in {args.image_dir}")

    dense_encoder = WaonSiglipEncoder()
    sparse_encoder = SpladeJapaneseSparseEncoder()
    ocr_text_extractor = PaddleOcrVlTextExtractor()
    image_captioner = Florence2Captioner()
    index = QdrantImageIndex(QdrantClient(url=args.qdrant_url), args.collection)

    for image_path in images:
        document = ingest_image_with_extracted_text(
            image_path=image_path,
            ocr_text_extractor=ocr_text_extractor,
            image_captioner=image_captioner,
            image_dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            image_index=index,
        )
        print(f"indexed\t{document.image_id.value}\t{document.image_path.value}")


if __name__ == "__main__":
    main()
