from __future__ import annotations

import os

import uvicorn

from picca_search.infrastructure.embedding_models import SpladeJapaneseSparseEncoder
from picca_search.services.sparse_api import create_sparse_app


def main() -> None:
    device = os.getenv("MODEL_DEVICE")
    port = int(os.getenv("PORT", "8002"))
    encoder = SpladeJapaneseSparseEncoder(device=device)
    uvicorn.run(create_sparse_app(encoder), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
