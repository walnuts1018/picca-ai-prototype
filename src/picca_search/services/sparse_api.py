from __future__ import annotations

from picca_search.infrastructure.embedding_models import SpladeJapaneseSparseEncoder
from picca_search.services.common import SparseVectorItem, SparseVectorsResponse, TextBatchRequest, create_service_app


def create_sparse_app(encoder: SpladeJapaneseSparseEncoder) -> object:
    app = create_service_app("sparse-service")

    @app.post("/encode/text-batch", response_model=SparseVectorsResponse)
    def encode_text_batch(request: TextBatchRequest) -> SparseVectorsResponse:
        vectors = encoder.encode_texts(request.texts)
        return SparseVectorsResponse(
            vectors=[
                SparseVectorItem(indices=list(vector.indices), values=list(vector.values))
                for vector in vectors
            ]
        )

    return app
