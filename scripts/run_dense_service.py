from __future__ import annotations

import os

import uvicorn

from picca_search.infrastructure.embedding_models import WaonSiglipEncoder
from picca_search.services.dense_api import create_dense_app


def main() -> None:
    device = os.getenv("MODEL_DEVICE")
    port = int(os.getenv("PORT", "8001"))
    encoder = WaonSiglipEncoder(device=device)
    uvicorn.run(create_dense_app(encoder), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
