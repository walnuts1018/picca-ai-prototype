from __future__ import annotations

from pathlib import Path
from typing import Protocol

from picca_search.domain import (
    DenseVector,
    ExtractedImageText,
    ImageDocument,
    ImageId,
    ImagePath,
    SearchQuery,
    SearchResult,
    SparseVector,
)


class ImageDenseEncoder(Protocol):
    def encode_image(self, image_path: Path) -> DenseVector: ...


class TextDenseEncoder(Protocol):
    def encode_text(self, text: str) -> DenseVector: ...


class SparseTextEncoder(Protocol):
    def encode_text(self, text: str) -> SparseVector: ...


class OcrTextExtractor(Protocol):
    def extract_text(self, image_path: Path) -> str: ...


class ImageCaptioner(Protocol):
    def caption(self, image_path: Path) -> str: ...


class ImageIndex(Protocol):
    def upsert(self, documents: list[ImageDocument]) -> None: ...

    def search(
        self,
        query_dense: DenseVector,
        query_sparse: SparseVector,
        limit: int,
    ) -> list[SearchResult]: ...


def ingest_image(
    image_path: Path,
    text: str,
    image_dense_encoder: ImageDenseEncoder,
    sparse_encoder: SparseTextEncoder,
    image_index: ImageIndex,
) -> ImageDocument:
    valid_path = ImagePath.create(image_path)
    normalized_text = text.strip()
    document = ImageDocument.create(
        image_id=ImageId.from_path(valid_path.value),
        image_path=valid_path,
        dense_vector=image_dense_encoder.encode_image(valid_path.value),
        sparse_vector=sparse_encoder.encode_text(normalized_text),
        text=normalized_text,
    )
    image_index.upsert([document])
    return document


def ingest_image_with_extracted_text(
    image_path: Path,
    ocr_text_extractor: OcrTextExtractor,
    image_captioner: ImageCaptioner,
    image_dense_encoder: ImageDenseEncoder,
    sparse_encoder: SparseTextEncoder,
    image_index: ImageIndex,
) -> ImageDocument:
    valid_path = ImagePath.create(image_path)
    extracted_text = ExtractedImageText.create(
        ocr_text=ocr_text_extractor.extract_text(valid_path.value),
        caption=image_captioner.caption(valid_path.value),
    )
    return ingest_image(
        image_path=valid_path.value,
        text=extracted_text.combined,
        image_dense_encoder=image_dense_encoder,
        sparse_encoder=sparse_encoder,
        image_index=image_index,
    )


def search_images(
    query_text: str,
    text_dense_encoder: TextDenseEncoder,
    sparse_encoder: SparseTextEncoder,
    image_index: ImageIndex,
    limit: int,
) -> list[SearchResult]:
    query = SearchQuery.create(query_text)
    if limit < 1:
        raise ValueError("Search limit must be greater than zero")
    query_dense = text_dense_encoder.encode_text(query.text)
    query_sparse = sparse_encoder.encode_text(query.text)
    return image_index.search(query_dense, query_sparse, limit)
