from __future__ import annotations

import argparse
import json

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the gateway search API.")
    parser.add_argument("query")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dense-weight", type=float)
    parser.add_argument("--ocr-weight", type=float)
    parser.add_argument("--florence-weight", type=float)
    parser.add_argument("--include-diagnostics", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = {
        "query": args.query,
        "limit": args.limit,
        "include_diagnostics": args.include_diagnostics,
    }
    if args.dense_weight is not None:
        payload["dense_weight"] = args.dense_weight
    if args.ocr_weight is not None:
        payload["ocr_weight"] = args.ocr_weight
    if args.florence_weight is not None:
        payload["florence_weight"] = args.florence_weight

    response = httpx.post(f"{args.base_url.rstrip('/')}/search", json=payload, timeout=300.0)
    response.raise_for_status()
    data = response.json()

    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    for result in data["results"]:
        print(f"{result['score']:.6f}\t{result['path']}\t{result['text']}")


if __name__ == "__main__":
    main()
