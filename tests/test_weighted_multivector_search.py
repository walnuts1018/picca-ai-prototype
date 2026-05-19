from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from qdrant_client import models

from picca_search.domain import DenseVector, ImageDocument, ImageId, ImagePath, SparseVector
from picca_search.infrastructure.qdrant_index import (
    DENSE_VECTOR_NAME,
    FLORENCE_SPARSE_VECTOR_NAME,
    OCR_SPARSE_VECTOR_NAME,
    QdrantImageIndex,
    point_from_document,
)


def _dense(values: list[float]) -> DenseVector:
    return DenseVector.create(values)


def _sparse(indices: list[int], values: list[float]) -> SparseVector:
    return SparseVector.create(indices, values)


TEST_IMAGE_PATH = ImagePath.create(Path("images/1727437432250.jpg"))


def test_point_from_document_separates_ocr_and_florence_sparse_vectors() -> None:
    document = ImageDocument.create(
        image_id=ImageId("img-1"),
        image_path=TEST_IMAGE_PATH,
        dense_vector=_dense([0.1, 0.2]),
        florence_sparse_vector=_sparse([1, 2], [0.3, 0.4]),
        text="caption only",
        ocr_sparse_vector=_sparse([7], [0.9]),
        ocr_text="receipt total",
        caption="store shelf",
    )

    point = point_from_document(document)

    assert point.vector[DENSE_VECTOR_NAME] == [0.1, 0.2]
    assert point.vector[OCR_SPARSE_VECTOR_NAME] == models.SparseVector(indices=[7], values=[0.9])
    assert point.vector[FLORENCE_SPARSE_VECTOR_NAME] == models.SparseVector(
        indices=[1, 2],
        values=[0.3, 0.4],
    )


def test_point_from_document_omits_ocr_sparse_vector_when_ocr_was_skipped() -> None:
    document = ImageDocument.create(
        image_id=ImageId("img-2"),
        image_path=TEST_IMAGE_PATH,
        dense_vector=_dense([0.1, 0.2]),
        florence_sparse_vector=_sparse([1], [0.3]),
        text="caption only",
        caption="mountain view",
    )

    point = point_from_document(document)

    assert OCR_SPARSE_VECTOR_NAME not in point.vector
    assert FLORENCE_SPARSE_VECTOR_NAME in point.vector


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        using = kwargs.get("using")
        if using == DENSE_VECTOR_NAME:
            return [
                SimpleNamespace(id="img-a", score=0.91, payload={"path": "a.jpg"}),
                SimpleNamespace(id="img-b", score=0.87, payload={"path": "b.jpg"}),
            ]
        if using == OCR_SPARSE_VECTOR_NAME:
            return [SimpleNamespace(id="img-a", score=1.3, payload={"path": "a.jpg"})]
        if using == FLORENCE_SPARSE_VECTOR_NAME:
            return [
                SimpleNamespace(id="img-b", score=1.1, payload={"path": "b.jpg"}),
                SimpleNamespace(id="img-a", score=0.9, payload={"path": "a.jpg"}),
            ]
        raise AssertionError(f"unexpected using={using!r}")


def test_search_with_diagnostics_uses_421_weights_and_zero_for_missing_ocr() -> None:
    index = QdrantImageIndex(_FakeClient(), "images")

    diagnostics = index.search_with_diagnostics(
        query_dense=_dense([0.5, 0.6]),
        query_ocr_sparse=_sparse([3], [0.7]),
        query_florence_sparse=_sparse([4], [0.8]),
        limit=5,
    )

    assert [result.image_id.value for result in diagnostics.fused] == ["img-a", "img-b"]
    assert diagnostics.fused[0].score == 4.0 / 2.0 + 2.0 / 2.0 + 1.0 / 3.0
    assert diagnostics.fused[1].score == 2.0 / 3.0 + 1.0 / 2.0
    assert [call.get("using") for call in index.client.calls] == [
        DENSE_VECTOR_NAME,
        OCR_SPARSE_VECTOR_NAME,
        FLORENCE_SPARSE_VECTOR_NAME,
    ]
