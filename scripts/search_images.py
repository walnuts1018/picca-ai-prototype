from __future__ import annotations

import argparse
import json

from qdrant_client import QdrantClient

from picca_search.application import search_images as run_search
from picca_search.infrastructure.embedding_models import (
    SpladeJapaneseSparseEncoder,
    WaonSiglipEncoder,
)
from picca_search.infrastructure.qdrant_index import (
    DENSE_WEIGHT,
    FLORENCE_WEIGHT,
    OCR_WEIGHT,
    RRF_K,
    QdrantImageIndex,
)

DEVICE_CHOICES = ("cuda", "mps", "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search indexed images with Japanese text.")
    parser.add_argument("query", help="Natural language search query.")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="picca_images")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dense-device", choices=DEVICE_CHOICES)
    parser.add_argument("--sparse-device", choices=DEVICE_CHOICES)
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print dense/ocr/florence ranks and weighted RRF contributions for each result.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit search diagnostics as JSON for evaluation workflows.",
    )
    args = parser.parse_args()

    dense_encoder = WaonSiglipEncoder(device=args.dense_device)
    sparse_encoder = SpladeJapaneseSparseEncoder(device=args.sparse_device)
    index = QdrantImageIndex(QdrantClient(url=args.qdrant_url), args.collection)

    if not args.explain and not args.json:
        results = run_search(
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
        return

    query_dense = dense_encoder.encode_text(args.query)
    query_sparse = sparse_encoder.encode_text(args.query)
    diagnostics = index.search_with_diagnostics(query_dense, query_sparse, query_sparse, args.limit)

    if args.json:
        def _encode_result(result):
            return {
                "image_id": result.image_id.value,
                "score": result.score,
                "payload": result.payload,
            }

        print(
            json.dumps(
                {
                    "query": args.query,
                    "limit": args.limit,
                    "dense": [_encode_result(result) for result in diagnostics.dense],
                    "ocr": [_encode_result(result) for result in diagnostics.ocr],
                    "florence": [_encode_result(result) for result in diagnostics.florence],
                    "fused": [_encode_result(result) for result in diagnostics.fused],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    dense_ranks = {result.image_id.value: result for result in diagnostics.dense}
    ocr_ranks = {result.image_id.value: result for result in diagnostics.ocr}
    florence_ranks = {result.image_id.value: result for result in diagnostics.florence}

    for result in diagnostics.fused:
        dense_result = dense_ranks.get(result.image_id.value)
        ocr_result = ocr_ranks.get(result.image_id.value)
        florence_result = florence_ranks.get(result.image_id.value)
        dense_score = DENSE_WEIGHT / (RRF_K + dense_result.rank) if dense_result is not None else 0.0
        ocr_score = OCR_WEIGHT / (RRF_K + ocr_result.rank) if ocr_result is not None else 0.0
        florence_score = (
            FLORENCE_WEIGHT / (RRF_K + florence_result.rank) if florence_result is not None else 0.0
        )
        if args.explain:
            print(
                f"{result.score:.6f}\t"
                f"dense={dense_score:.6f}@{dense_result.rank if dense_result else '-'}\t"
                f"ocr={ocr_score:.6f}@{ocr_result.rank if ocr_result else '-'}\t"
                f"florence={florence_score:.6f}@{florence_result.rank if florence_result else '-'}\t"
                f"{result.payload.get('path', '')}\t"
                f"{result.payload.get('text', '')}"
            )
        else:
            print(
                f"{result.score:.6f}\t{result.payload.get('path', '')}\t"
                f"{result.payload.get('text', '')}"
            )


if __name__ == "__main__":
    main()
