# picca-ai-prototype

自然言語画像検索プロトタイプです。現行構成は script 実行型ではなく、`gateway + model services + RabbitMQ + SeaweedFS + Qdrant` の常駐サービス構成です。

## Services

- `gateway`: 検索 HTTP API と RabbitMQ consumer
- `dense-service`: WAON SigLIP
- `sparse-service`: light-SPLADE Japanese
- `ocr-service`: PaddleOCR-VL
- `caption-service`: Florence-2 + CAT-Translate

- `rabbitmq`: image job queue
- `seaweedfs`: S3 互換オブジェクトストレージ
- `qdrant`: ベクトル検索

## Local Run

```bash
# モデルのダウンロードと ONNX エクスポートを事前に実行
uv run scripts/prepare_models.py --output-dir models/

# 起動
docker compose up --build
```

デフォルトの Compose は、ローカルの `./models` ディレクトリをコンテナにマウントし、ONNX モデル（存在する場合）またはキャッシュされた PyTorch モデルを使用します。これにより、起動時のダウンロードを回避し、推論を高速化します。

全モデルを `MODEL_DEVICE=cpu` で起動します。モデル単位で CUDA に切り替えたい場合は、対象サービスの image / Dockerfile を `model-cuda.Dockerfile` ベースに差し替え、`MODEL_DEVICE=cuda` を指定してください。

## ONNX Integration

標準的な Hugging Face モデル（SigLIP, SPLADE, CAT-Translate）は ONNX Runtime で動作します。
`scripts/prepare_models.py` は以下の処理を行い、全て `./models` ディレクトリに集約します：
- **ONNX Export:** SigLIP, SPLADE, CAT-Translate を ONNX 形式で保存。
- **Local Caching:** Florence-2 を PyTorch 形式、PaddleOCR を PaddleX 形式 (`./models/paddlex`) で保存。

インフラ層は、指定ディレクトリに `.onnx` ファイルがあれば自動的に ONNX Runtime を使用し、なければ PyTorch にフォールバックします。PaddleOCR は `PADDLEX_HOME` 環境変数を通じて `./models/paddlex` を参照します。

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
