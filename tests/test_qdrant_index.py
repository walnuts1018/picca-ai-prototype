from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from qdrant_client import models

from picca_search.domain import DenseVector, ImageDocument, ImageId, ImagePath, SparseVector
from picca_search.infrastructure.qdrant_index import QdrantImageIndex


class RecordingClient:
    def __init__(self) -> None:
        self.collection_exists_calls = 0
        self.create_collection_calls: list[dict[str, object]] = []
        self.upsert_calls: list[dict[str, object]] = []
        self.query_points_calls: list[dict[str, object]] = []

    def collection_exists(self, collection_name: str) -> bool:
        self.collection_exists_calls += 1
        return False

    def create_collection(self, **kwargs) -> None:
        self.create_collection_calls.append(kwargs)

    def upsert(self, **kwargs) -> None:
        self.upsert_calls.append(kwargs)

    def query_points(self, **kwargs):
        self.query_points_calls.append(kwargs)
        return SimpleNamespace(
            points=[
                models.ScoredPoint(
                    id="image-1",
                    version=1,
                    score=0.5,
                    payload={"path": "/tmp/a.jpg", "text": "a"},
                    vector=None,
                    shard_key=None,
                    order_value=None,
                )
            ]
        )


def make_document(path: Path) -> ImageDocument:
    path.write_bytes(b"fake")
    return ImageDocument.create(
        image_id=ImageId.from_path(path),
        image_path=ImagePath.create(path),
        dense_vector=DenseVector.create([0.1, 0.2]),
        sparse_vector=SparseVector.create([1, 2], [0.3, 0.4]),
        text="hello",
    )


def test_upsert_checks_collection_only_once(tmp_path: Path) -> None:
    client = RecordingClient()
    index = QdrantImageIndex(client, "images")

    first = make_document(tmp_path / "first.jpg")
    second = make_document(tmp_path / "second.jpg")

    index.upsert([first])
    index.upsert([second])

    assert client.collection_exists_calls == 1
    assert len(client.create_collection_calls) == 1
    assert len(client.upsert_calls) == 2


def test_search_uses_single_fusion_query(tmp_path: Path) -> None:
    client = RecordingClient()
    index = QdrantImageIndex(client, "images")

    _ = tmp_path
    results = index.search(
        DenseVector.create([0.1, 0.2]),
        SparseVector.create([1, 2], [0.3, 0.4]),
        limit=3,
    )

    assert [result.image_id.value for result in results] == ["image-1"]
    assert len(client.query_points_calls) == 1
    query = client.query_points_calls[0]
    assert query["collection_name"] == "images"
    assert isinstance(query["query"], models.FusionQuery)
    assert len(query["prefetch"]) == 2


def test_search_with_diagnostics_keeps_per_source_rankings() -> None:
    client = RecordingClient()
    index = QdrantImageIndex(client, "images")

    diagnostics = index.search_with_diagnostics(
        DenseVector.create([0.1, 0.2]),
        SparseVector.create([1, 2], [0.3, 0.4]),
        limit=3,
    )

    assert len(client.query_points_calls) == 2
    assert diagnostics.dense[0].rank == 1
    assert diagnostics.sparse[0].rank == 1
    assert diagnostics.fused[0].image_id.value == "image-1"
