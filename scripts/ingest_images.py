from __future__ import annotations

import argparse
from dataclasses import dataclass
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageOps
from qdrant_client import QdrantClient

from picca_search.application import build_image_document_with_extracted_text
from picca_search.domain import ExtractedImageText, ImageDocument, ImageId, ImagePath
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

DEVICE_CHOICES = ("cuda", "mps", "cpu")


@contextmanager
def prepare_inference_image(image_path: Path) -> Iterator[Path]:
    with Image.open(image_path) as image:
        normalized_image = ImageOps.exif_transpose(image)
        width, height = normalized_image.size
        long_edge = max(width, height)
        needs_orientation_normalization = _needs_orientation_normalization(image)
        if long_edge <= 2048 and not needs_orientation_normalization:
            yield image_path
            return

        output_image = normalized_image
        if long_edge > 2048:
            scale = 2048 / long_edge
            resized_size = (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            )
            output_image = normalized_image.resize(resized_size, Image.Resampling.LANCZOS)

        suffix, output_format = _select_output_format(image_path, normalized_image)

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)

        try:
            if output_format == "PNG":
                output_image.save(temporary_path, format=output_format)
            else:
                output_image.convert("RGB").save(
                    temporary_path,
                    format=output_format,
                    quality=90,
                    optimize=True,
                )
            yield temporary_path
        finally:
            temporary_path.unlink(missing_ok=True)


def _needs_orientation_normalization(image: Image.Image) -> bool:
    orientation = image.getexif().get(274, 1)
    return orientation != 1


def _select_output_format(image_path: Path, image: Image.Image) -> tuple[str, str]:
    has_alpha = "A" in image.getbands() or image.info.get("transparency") is not None
    if has_alpha or image_path.suffix.lower() in {".png", ".bmp"}:
        return ".png", "PNG"

    return ".jpg", "JPEG"


@dataclass
class _PendingImage:
    image_path: Path
    image: Image.Image
    extracted_text: ExtractedImageText


class IngestionBatchAccumulator:
    MAX_BATCH_SIZE = 64

    def __init__(
        self,
        *,
        image_dense_encoder: WaonSiglipEncoder,
        sparse_encoder: SpladeJapaneseSparseEncoder,
        batch_size: int,
    ):
        if batch_size > self.MAX_BATCH_SIZE:
            raise ValueError(
                f"batch_size={batch_size} exceeds MAX_BATCH_SIZE={self.MAX_BATCH_SIZE}. "
                f"Large batches can cause OOM due to in-memory PIL.Image storage."
            )
        self.encoder = image_dense_encoder
        self.sparse = sparse_encoder
        self.batch_size = batch_size
        self.pending: list[_PendingImage] = []

    def add(self, image_path: Path, image: Image.Image, ocr_text: str, caption: str) -> None:
        extracted_text = ExtractedImageText.create(ocr_text, caption)
        self.pending.append(_PendingImage(image_path, image, extracted_text))

    def is_ready(self) -> bool:
        return len(self.pending) >= self.batch_size

    def flush(self) -> list[ImageDocument]:
        if not self.pending:
            return []
        pending = self.pending
        self.pending = []
        images = [p.image for p in pending]
        texts = [p.extracted_text.combined for p in pending]
        dense_vectors = self.encoder.encode_images(images)
        sparse_vectors = self.sparse.encode_texts(texts)
        documents: list[ImageDocument] = []
        for p, dense, sparse, text in zip(pending, dense_vectors, sparse_vectors, texts):
            doc = ImageDocument.create(
                image_id=ImageId.from_path(p.image_path),
                image_path=ImagePath.create(p.image_path),
                dense_vector=dense,
                sparse_vector=sparse,
                text=text,
                ocr_text=p.extracted_text.ocr_text,
                caption=p.extracted_text.caption,
            )
            documents.append(doc)
        return documents


def ingest_image(
    *,
    image_path: Path,
    ocr_text_extractor: PaddleOcrVlTextExtractor,
    image_captioner: Florence2Captioner,
    image_dense_encoder: WaonSiglipEncoder,
    sparse_encoder: SpladeJapaneseSparseEncoder,
    image_index: QdrantImageIndex,
):
    with prepare_inference_image(image_path) as inference_image_path:
        return build_image_document_with_extracted_text(
            image_path=image_path,
            inference_image_path=inference_image_path,
            ocr_text_extractor=ocr_text_extractor,
            image_captioner=image_captioner,
            image_dense_encoder=image_dense_encoder,
            sparse_encoder=sparse_encoder,
        )


def ingest_images(
    *,
    image_paths: list[Path],
    ocr_text_extractor: PaddleOcrVlTextExtractor,
    image_captioner: Florence2Captioner,
    image_dense_encoder: WaonSiglipEncoder,
    sparse_encoder: SpladeJapaneseSparseEncoder,
    image_index: QdrantImageIndex,
    batch_size: int,
) -> list[ImageDocument]:
    if batch_size < 1:
        raise ValueError("Batch size must be greater than zero")

    accumulator = IngestionBatchAccumulator(
        image_dense_encoder=image_dense_encoder,
        sparse_encoder=sparse_encoder,
        batch_size=batch_size,
    )
    documents: list[ImageDocument] = []

    for image_path in image_paths:
        with prepare_inference_image(image_path) as inference_path:
            ocr_text = ocr_text_extractor.extract_text(inference_path)
            caption = image_captioner.caption(inference_path)
            with Image.open(inference_path).convert("RGB") as img:
                accumulator.add(image_path, img.copy(), ocr_text, caption)

        if accumulator.is_ready():
            batch_docs = accumulator.flush()
            image_index.upsert(batch_docs)
            documents.extend(batch_docs)

    remaining = accumulator.flush()
    if remaining:
        image_index.upsert(remaining)
        documents.extend(remaining)

    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local images into Qdrant.")
    parser.add_argument("image_dir", type=Path, help="Directory containing image files.")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="picca_images")
    parser.add_argument("--dense-device", choices=DEVICE_CHOICES)
    parser.add_argument("--sparse-device", choices=DEVICE_CHOICES)
    parser.add_argument("--caption-device", choices=DEVICE_CHOICES)
    parser.add_argument("--ocr-device", choices=DEVICE_CHOICES)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    images = sorted(
        path
        for path in args.image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if len(images) == 0:
        raise SystemExit(f"No supported images found in {args.image_dir}")

    dense_encoder = WaonSiglipEncoder(device=args.dense_device)
    sparse_encoder = SpladeJapaneseSparseEncoder(device=args.sparse_device)
    ocr_text_extractor = PaddleOcrVlTextExtractor(device=args.ocr_device)
    image_captioner = Florence2Captioner(device=args.caption_device)
    index = QdrantImageIndex(QdrantClient(url=args.qdrant_url), args.collection)

    documents = ingest_images(
        image_paths=images,
        ocr_text_extractor=ocr_text_extractor,
        image_captioner=image_captioner,
        image_dense_encoder=dense_encoder,
        sparse_encoder=sparse_encoder,
        image_index=index,
        batch_size=args.batch_size,
    )
    for document in documents:
        print(f"indexed\t{document.image_id.value}\t{document.image_path.value}")


if __name__ == "__main__":
    main()
