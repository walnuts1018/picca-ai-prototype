from __future__ import annotations

from qdrant_client import QdrantClient, models

from picca_search.domain import DenseVector, ImageDocument, ImageId, SearchResult, SparseVector


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


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


class QdrantImageIndex:
    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self.client = client
        self.collection_name = collection_name

    def ensure_collection(self, dense_vector_size: int) -> None:
        if self.client.collection_exists(self.collection_name):
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
