from __future__ import annotations

from fastapi.testclient import TestClient

from picca_search.domain import DenseVector, ImageId, SearchResult, SparseVector
from picca_search.gateway.search_api import GatewaySearchDependencies, create_search_app


class _FakeDense:
    def encode_texts(self, texts: list[str]) -> list[DenseVector]:
        return [DenseVector.create([1.0, 0.0])]


class _FakeSparse:
    def encode_texts(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector.create([1], [1.0])]


class _FakeIndex:
    def search(self, **kwargs: object) -> list[SearchResult]:
        payload = {"path": "s3://images/a.jpg", "text": "torii", "caption": "red gate"}
        return [SearchResult(image_id=ImageId("a.jpg"), score=3.5, payload=payload)]

    def search_with_diagnostics(self, **kwargs: object) -> object:
        class _Ranked:
            def __init__(self, image_id: str, rank: int, score: float, payload: dict[str, object]) -> None:
                self.image_id = ImageId(image_id)
                self.rank = rank
                self.score = score
                self.payload = payload

        payload = {"path": "s3://images/a.jpg", "text": "torii", "caption": "red gate"}
        fused = [SearchResult(image_id=ImageId("a.jpg"), score=3.5, payload=payload)]
        dense = [_Ranked("a.jpg", 1, 0.91, payload)]
        ocr = [_Ranked("a.jpg", 2, 0.52, payload)]
        florence: list[_Ranked] = []
        return type(
            "Diagnostics",
            (),
            {"fused": fused, "dense": dense, "ocr": ocr, "florence": florence},
        )()


def test_search_with_diagnostics_returns_score_breakdown() -> None:
    app = create_search_app(
        GatewaySearchDependencies(
            dense_client=_FakeDense(),
            sparse_client=_FakeSparse(),
            index=_FakeIndex(),
        )
    )
    client = TestClient(app)

    response = client.post("/search", json={"query": "torii", "include_diagnostics": True})

    assert response.status_code == 200
    body = response.json()
    result = body["results"][0]
    assert result["score"] == 3.5
    assert result["score_breakdown"]["dense_rank"] == 1
    assert result["score_breakdown"]["ocr_rank"] == 2
    assert result["score_breakdown"]["florence_rank"] is None
    assert result["score_breakdown"]["dense_score"] == 2.0
    assert result["score_breakdown"]["ocr_score"] == 2.0 / 3.0
    assert result["score_breakdown"]["florence_score"] == 0.0


def test_search_without_diagnostics_keeps_existing_response_shape() -> None:
    app = create_search_app(
        GatewaySearchDependencies(
            dense_client=_FakeDense(),
            sparse_client=_FakeSparse(),
            index=_FakeIndex(),
        )
    )
    client = TestClient(app)

    response = client.post("/search", json={"query": "torii"})

    assert response.status_code == 200
    body = response.json()
    assert body["diagnostics"] is None
    result = body["results"][0]
    assert result["image_id"] == "a.jpg"
    assert result["score"] == 3.5
    assert result["path"] == "s3://images/a.jpg"
    assert result["text"] == "torii"
    assert "score_breakdown" not in result
