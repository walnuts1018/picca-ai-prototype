from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from picca_search.domain import DenseVector, ImageDocument, ImageId, SearchResult, SparseVector


DENSE_VECTOR_NAME = "dense"
OCR_SPARSE_VECTOR_NAME = "ocr_sparse"
FLORENCE_SPARSE_VECTOR_NAME = "florence_sparse"
DENSE_WEIGHT = 4.0
OCR_WEIGHT = 2.0
FLORENCE_WEIGHT = 1.0
RRF_K = 1
RRF_WEIGHTS = (DENSE_WEIGHT, OCR_WEIGHT, FLORENCE_WEIGHT)
DEFAULT_RRF_WEIGHTS = RRF_WEIGHTS


@dataclass(frozen=True)
class RankedSearchResult:
    image_id: ImageId
    score: float
    payload: dict[str, object]
    rank: int


@dataclass(frozen=True)
class SearchDiagnostics:
    fused: list[SearchResult]
    dense: list[RankedSearchResult]
    ocr: list[RankedSearchResult]
    florence: list[RankedSearchResult]


def point_from_document(document: ImageDocument) -> models.PointStruct:
    vector: dict[str, list[float] | models.SparseVector] = {
        DENSE_VECTOR_NAME: list(document.dense_vector.values),
        FLORENCE_SPARSE_VECTOR_NAME: models.SparseVector(
            indices=list(document.florence_sparse_vector.indices),
            values=list(document.florence_sparse_vector.values),
        ),
    }
    if document.ocr_sparse_vector is not None:
        vector[OCR_SPARSE_VECTOR_NAME] = models.SparseVector(
            indices=list(document.ocr_sparse_vector.indices),
            values=list(document.ocr_sparse_vector.values),
        )
    return models.PointStruct(
        id=document.image_id.value,
        vector=vector,
        payload=document.payload,
    )


def prefetches_from_query(
    query_dense: DenseVector,
    query_ocr_sparse: SparseVector,
    query_florence_sparse: SparseVector,
    limit: int,
) -> list[models.Prefetch]:
    return [
        models.Prefetch(
            query=list(query_dense.values),
            using=DENSE_VECTOR_NAME,
            limit=limit,
        ),
        models.Prefetch(
            query=models.SparseVector(
                indices=list(query_ocr_sparse.indices),
                values=list(query_ocr_sparse.values),
            ),
            using=OCR_SPARSE_VECTOR_NAME,
            limit=limit,
        ),
        models.Prefetch(
            query=models.SparseVector(
                indices=list(query_florence_sparse.indices),
                values=list(query_florence_sparse.values),
            ),
            using=FLORENCE_SPARSE_VECTOR_NAME,
            limit=limit,
        ),
    ]


def search_result_from_scored_point(point: models.ScoredPoint) -> SearchResult:
    return SearchResult(
        image_id=ImageId(str(point.id)),
        score=float(point.score),
        payload=dict(point.payload or {}),
    )


def ranked_result_from_scored_point(point: models.ScoredPoint, rank: int) -> RankedSearchResult:
    return RankedSearchResult(
        image_id=ImageId(str(point.id)),
        score=float(point.score),
        payload=dict(point.payload or {}),
        rank=rank,
    )


class QdrantImageIndex:
    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self.client = client
        self.collection_name = collection_name
        self._collection_ready = False

    def ensure_collection(self, dense_vector_size: int) -> None:
        if self._collection_ready:
            return
        if self.client.collection_exists(self.collection_name):
            self._collection_ready = True
            return
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=dense_vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        always_ram=True,
                    )
                ),
                sparse_vectors_config={
                    OCR_SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    ),
                    FLORENCE_SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                self._collection_ready = True
                return
            raise
        self._collection_ready = True

    def upsert(self, documents: list[ImageDocument]) -> None:
        if len(documents) == 0:
            return
        self.ensure_collection(documents[0].dense_vector.dimension)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point_from_document(document) for document in documents],
        )

    def search(
        self,
        query_dense: DenseVector,
        query_ocr_sparse: SparseVector,
        query_florence_sparse: SparseVector,
        limit: int,
        weights: tuple[float, float, float] | None = None,
    ) -> list[SearchResult]:
        self.ensure_collection(query_dense.dimension)
        resolved_weights = DEFAULT_RRF_WEIGHTS if weights is None else weights
        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetches_from_query(query_dense, query_ocr_sparse, query_florence_sparse, limit),
            query=models.RrfQuery(
                rrf=models.Rrf(
                    k=RRF_K,
                    weights=list(resolved_weights),
                )
            ),
            limit=limit,
            with_payload=True,
        )
        points = getattr(response, "points", response)
        return [search_result_from_scored_point(point) for point in points]

    def search_with_diagnostics(
        self,
        query_dense: DenseVector,
        query_ocr_sparse: SparseVector,
        query_florence_sparse: SparseVector,
        limit: int,
        weights: tuple[float, float, float] | None = None,
    ) -> SearchDiagnostics:
        self.ensure_collection(query_dense.dimension)
        resolved_weights = DEFAULT_RRF_WEIGHTS if weights is None else weights
        dense_response = self.client.query_points(
            collection_name=self.collection_name,
            query=list(query_dense.values),
            using=DENSE_VECTOR_NAME,
            limit=limit,
            with_payload=True,
        )
        ocr_response = self.client.query_points(
            collection_name=self.collection_name,
            query=models.SparseVector(
                indices=list(query_ocr_sparse.indices),
                values=list(query_ocr_sparse.values),
            ),
            using=OCR_SPARSE_VECTOR_NAME,
            limit=limit,
            with_payload=True,
        )
        florence_response = self.client.query_points(
            collection_name=self.collection_name,
            query=models.SparseVector(
                indices=list(query_florence_sparse.indices),
                values=list(query_florence_sparse.values),
            ),
            using=FLORENCE_SPARSE_VECTOR_NAME,
            limit=limit,
            with_payload=True,
        )
        dense_points = getattr(dense_response, "points", dense_response)
        ocr_points = getattr(ocr_response, "points", ocr_response)
        florence_points = getattr(florence_response, "points", florence_response)
        dense = [ranked_result_from_scored_point(point, rank=idx + 1) for idx, point in enumerate(dense_points)]
        ocr = [ranked_result_from_scored_point(point, rank=idx + 1) for idx, point in enumerate(ocr_points)]
        florence = [
            ranked_result_from_scored_point(point, rank=idx + 1)
            for idx, point in enumerate(florence_points)
        ]

        fused_scores: dict[str, SearchResult] = {}
        for results, weight in zip((dense, ocr, florence), resolved_weights, strict=True):
            for result in results:
                existing = fused_scores.get(result.image_id.value)
                contribution = weight / (RRF_K + result.rank)
                if existing is None:
                    fused_scores[result.image_id.value] = SearchResult(
                        image_id=result.image_id,
                        score=contribution,
                        payload=result.payload,
                    )
                else:
                    fused_scores[result.image_id.value] = SearchResult(
                        image_id=existing.image_id,
                        score=existing.score + contribution,
                        payload=existing.payload,
                    )

        fused = sorted(
            fused_scores.values(),
            key=lambda result: result.score,
            reverse=True,
        )[:limit]
        return SearchDiagnostics(fused=fused, dense=dense, ocr=ocr, florence=florence)
