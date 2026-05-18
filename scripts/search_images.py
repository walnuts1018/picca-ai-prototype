from __future__ import annotations

import argparse

from qdrant_client import QdrantClient

from picca_search.application import search_images
from picca_search.infrastructure.embedding_models import (
    SpladeJapaneseSparseEncoder,
    WaonSiglipEncoder,
)
from picca_search.infrastructure.qdrant_index import QdrantImageIndex

DEVICE_CHOICES = ("cuda", "mps", "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search indexed images with Japanese text.")
    parser.add_argument("query", help="Natural language search query.")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="picca_images")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dense-device", choices=DEVICE_CHOICES)
    parser.add_argument("--sparse-device", choices=DEVICE_CHOICES)
    args = parser.parse_args()

    dense_encoder = WaonSiglipEncoder(device=args.dense_device)
    sparse_encoder = SpladeJapaneseSparseEncoder(device=args.sparse_device)
    index = QdrantImageIndex(QdrantClient(url=args.qdrant_url), args.collection)

    results = search_images(
        query_text=args.query,
        text_dense_encoder=dense_encoder,
        sparse_encoder=sparse_encoder,
        image_index=index,
        limit=args.limit,
    )
    for result in results:
        print(
            f"{result.score:.6f}\t{result.payload.get('path', '')}\t"
            f"{result.payload.get('text', '')}"
        )


if __name__ == "__main__":
    main()
