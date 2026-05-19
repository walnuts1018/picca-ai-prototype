# picca-ai-prototype

自然言語画像検索プロトタイプです。現行構成は script 実行型ではなく、`gateway + model services + RabbitMQ + SeaweedFS + Qdrant` の常駐サービス構成です。

## Services

- `gateway`: 検索 HTTP API と RabbitMQ consumer
- `dense-service`: WAON SigLIP
- `sparse-service`: light-SPLADE Japanese
- `ocr-service`: PaddleOCR-VL
- `caption-service`: Florence-2 + CAT-Translate
- `rabbitmq`: image job queue
- `seaweedfs-s3`: S3 互換オブジェクトストレージ
- `qdrant`: ベクトル検索

## Local Run

```bash
docker compose up --build
```

デフォルトの Compose は全モデルを `MODEL_DEVICE=cpu` で起動します。モデル単位で CUDA に切り替えたい場合は、対象サービスの image / Dockerfile を `Dockerfile.model.cuda` ベースに差し替え、`MODEL_DEVICE=cuda` を指定してください。

## Upload + Publish

ディレクトリ内画像を SeaweedFS S3 に配置し、その object key を RabbitMQ に流します。

```bash
uv run python scripts/publish_directory_to_queue.py ./images
```

## Search API

```bash
uv run python scripts/search_api_client.py "赤い鳥居が写っている写真" --limit 5
```

`dense_weight`, `ocr_weight`, `florence_weight`, `limit` は optional に指定できます。

## Runtime Notes

- `image_id` は S3 object key をそのまま使います
- `gateway` は image job を batch 収集し、SeaweedFS から画像を取得して取り込みます
- dense / sparse は batch endpoint でまとめて推論します
- OCR / caption は画像ごとに service を呼びます

## Entry Points

- `scripts/run_gateway.py`
- `scripts/run_dense_service.py`
- `scripts/run_sparse_service.py`
- `scripts/run_ocr_service.py`
- `scripts/run_caption_service.py`
