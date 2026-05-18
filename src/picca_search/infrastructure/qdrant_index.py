from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from picca_search.domain import DenseVector, ImageDocument, ImageId, SearchResult, SparseVector


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
RRF_K = 1


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
    sparse: list[RankedSearchResult]


def point_from_document(document: ImageDocument) -> models.PointStruct:
    return models.PointStruct(
        id=document.image_id.value,
        vector={
            DENSE_VECTOR_NAME: list(document.dense_vector.values),
            SPARSE_VECTOR_NAME: models.SparseVector(
                indices=list(document.sparse_vector.indices),
                values=list(document.sparse_vector.values),
            ),
        },
        payload=document.payload,
    )


def prefetches_from_query(
    query_dense: DenseVector,
    query_sparse: SparseVector,
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
                indices=list(query_sparse.indices),
                values=list(query_sparse.values),
            ),
            using=SPARSE_VECTOR_NAME,
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
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=dense_vector_size,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )
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
        query_sparse: SparseVector,
        limit: int,
    ) -> list[SearchResult]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetches_from_query(query_dense, query_sparse, limit),
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        points = getattr(response, "points", response)
        return [search_result_from_scored_point(point) for point in points]

    def search_with_diagnostics(
        self,
        query_dense: DenseVector,
        query_sparse: SparseVector,
        limit: int,
    ) -> SearchDiagnostics:
        dense_response = self.client.query_points(
            collection_name=self.collection_name,
            query=list(query_dense.values),
            using=DENSE_VECTOR_NAME,
            limit=limit,
            with_payload=True,
        )
        sparse_response = self.client.query_points(
            collection_name=self.collection_name,
            query=models.SparseVector(
                indices=list(query_sparse.indices),
                values=list(query_sparse.values),
            ),
            using=SPARSE_VECTOR_NAME,
            limit=limit,
            with_payload=True,
        )
        dense_points = getattr(dense_response, "points", dense_response)
        sparse_points = getattr(sparse_response, "points", sparse_response)
        dense = [ranked_result_from_scored_point(point, rank=idx + 1) for idx, point in enumerate(dense_points)]
        sparse = [ranked_result_from_scored_point(point, rank=idx + 1) for idx, point in enumerate(sparse_points)]

        fused_scores: dict[str, SearchResult] = {}
        for result in dense:
            fused_scores[result.image_id.value] = SearchResult(
                image_id=result.image_id,
                score=1.0 / (RRF_K + result.rank),
                payload=result.payload,
            )
        for result in sparse:
            existing = fused_scores.get(result.image_id.value)
            contribution = 1.0 / (RRF_K + result.rank)
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
        return SearchDiagnostics(fused=fused, dense=dense, sparse=sparse)
