from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI

from picca_search.domain import SearchQuery
from picca_search.gateway.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResultPayload,
    SearchScoreBreakdown,
)
from picca_search.infrastructure.model_client import DenseModelClient, SparseModelClient
from picca_search.infrastructure.qdrant_index import DEFAULT_RRF_WEIGHTS, QdrantImageIndex, RRF_K


@dataclass(frozen=True)
class GatewaySearchDependencies:
    dense_client: DenseModelClient
    sparse_client: SparseModelClient
    index: QdrantImageIndex


def create_search_app(dependencies: GatewaySearchDependencies) -> FastAPI:
    app = FastAPI(title="picca-gateway")
    app.state.gateway_dependencies = dependencies

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/search", response_model=SearchResponse, response_model_exclude_unset=True)
    def search(request: SearchRequest) -> SearchResponse:
        query = SearchQuery.create(request.query)
        dense_vector = dependencies.dense_client.encode_texts([query.text])[0]
        sparse_vector = dependencies.sparse_client.encode_texts([query.text])[0]
        weights = request.weights_tuple(DEFAULT_RRF_WEIGHTS)
        if request.include_diagnostics:
            diagnostics = dependencies.index.search_with_diagnostics(
                query_dense=dense_vector,
                query_ocr_sparse=sparse_vector,
                query_florence_sparse=sparse_vector,
                limit=request.limit,
                weights=weights,
            )
            breakdowns = _build_breakdowns(diagnostics, weights)
            return SearchResponse(
                query=query.text,
                limit=request.limit,
                weights=_weight_dict(weights),
                results=[
                    _result_payload(result, score_breakdown=breakdowns.get(result.image_id.value))
                    for result in diagnostics.fused
                ],
                diagnostics={
                    "dense": [_result_payload(result) for result in diagnostics.dense],
                    "ocr": [_result_payload(result) for result in diagnostics.ocr],
                    "florence": [_result_payload(result) for result in diagnostics.florence],
                },
            )

        results = dependencies.index.search(
            query_dense=dense_vector,
            query_ocr_sparse=sparse_vector,
            query_florence_sparse=sparse_vector,
            limit=request.limit,
            weights=weights,
        )
        return SearchResponse(
            query=query.text,
            limit=request.limit,
            weights=_weight_dict(weights),
            results=[_result_payload(result) for result in results],
            diagnostics=None,
        )

    return app


def _weight_dict(weights: tuple[float, float, float]) -> dict[str, float]:
    return {"dense": weights[0], "ocr": weights[1], "florence": weights[2]}


def _result_payload(
    result: Any,
    score_breakdown: SearchScoreBreakdown | None = None,
) -> SearchResultPayload:
    payload = dict(result.payload)
    result_payload: dict[str, object] = {
        "image_id": result.image_id.value,
        "score": result.score,
        "path": str(payload.get("path", "")),
        "text": str(payload.get("text", "")),
        "ocr_text": (str(payload["ocr_text"]) if "ocr_text" in payload else None),
        "caption": (str(payload["caption"]) if "caption" in payload else None),
    }
    if score_breakdown is not None:
        result_payload["score_breakdown"] = score_breakdown
    return SearchResultPayload(
        **result_payload,
    )


def _build_breakdowns(
    diagnostics: Any,
    weights: tuple[float, float, float],
) -> dict[str, SearchScoreBreakdown]:
    breakdowns: dict[str, dict[str, float | int | None]] = {}
    for source, results, weight in (
        ("dense", diagnostics.dense, weights[0]),
        ("ocr", diagnostics.ocr, weights[1]),
        ("florence", diagnostics.florence, weights[2]),
    ):
        for result in results:
            image_breakdown = breakdowns.setdefault(
                result.image_id.value,
                {
                    "dense_score": 0.0,
                    "ocr_score": 0.0,
                    "florence_score": 0.0,
                    "dense_rank": None,
                    "ocr_rank": None,
                    "florence_rank": None,
                },
            )
            image_breakdown[f"{source}_rank"] = result.rank
            image_breakdown[f"{source}_score"] = weight / (RRF_K + result.rank)
    return {
        image_id: SearchScoreBreakdown(**score_breakdown)
        for image_id, score_breakdown in breakdowns.items()
    }
