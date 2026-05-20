# picca-ai-prototype

自然言語画像検索プロトタイプです。現行構成は script 実行型ではなく、`gateway + model services + RabbitMQ + SeaweedFS + Qdrant` の常駐サービス構成です。

## Services

- `gateway`: 検索 HTTP API と RabbitMQ consumer
- `debug-web`: debug 用の upload/search UI、sqlite status tracker、S3 image proxy
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

# 生成済みモデルを OCI Artifact として push
export GITHUB_ACTOR=your-username
export GITHUB_TOKEN=your-token
go run ./scripts/artifactor --dir models --repo ghcr.io/your-org/picca-models --tag latest

# 起動
docker compose up --build
```

デフォルトの Compose は、ローカルの `./models` ディレクトリをコンテナにマウントし、事前に保存したローカルモデルだけを使用します。これにより、起動時のモデルダウンロードを回避します。

`scripts/artifactor` は `./models` 配下のトップレベルエントリを `linux/amd64` 固定の単一 OCI manifest として registry に push します。トップレベルのディレクトリは一時的に `tar.gz` layer 化し、通常ファイルはそのまま layer として扱います。認証情報は `--username` / `--password` または `OCI_REGISTRY_USERNAME` / `OCI_REGISTRY_PASSWORD`、GHCR の場合は `GITHUB_ACTOR` / `GITHUB_TOKEN` も利用できます。

## Published Images

GitHub Actions は `main` push または `workflow_dispatch` で以下を GHCR に publish します。

- `ghcr.io/<owner>/picca-ai-prototype-gateway`: `linux/amd64` + `linux/arm64`
- `ghcr.io/<owner>/picca-ai-prototype-model`: `linux/amd64`
- `ghcr.io/<owner>/picca-ai-prototype-model-cuda`: `linux/amd64`

## Published Model Artifact

GitHub Actions の `Publish Model Artifact` workflow は `workflow_dispatch` でのみ実行され、`HF_TOKEN` secret を使ってモデルをダウンロードしてから、以下へ OCI Artifact として push します。

- `ghcr.io/<owner>/picca-ai-prototype-models:<image_tag>`

manual 実行時は `image_tag` input が必須で、既定値は `dev` です。

全モデルを `MODEL_DEVICE=cpu` で起動します。モデル単位で CUDA に切り替えたい場合は、対象サービスの image / Dockerfile を `model-cuda.Dockerfile` ベースに差し替え、`MODEL_DEVICE=cuda` を指定してください。

## ONNX Integration

標準的な Hugging Face モデル（SigLIP, SPLADE, CAT-Translate）は ONNX Runtime で動作します。
`scripts/prepare_models.py` は以下の処理を行います：

- **ONNX Export:** SigLIP, SPLADE, CAT-Translate を ONNX 形式で `./models` に保存。
- **Paddle OCR Local Files:** PaddleOCR 用の `PP-OCRv5_mobile_det`、`PP-DocLayoutV2`、`PaddleOCR-VL` を `./models/paddlex/official_models` に保存します。
- **Local Caching:** Florence-2 を PyTorch 形式で `./models` に保存。

インフラ層は、指定ディレクトリに `.onnx` ファイルがあれば自動的に ONNX Runtime を使用します。PaddleOCR は `PADDLEX_HOME=/models/paddlex` 配下のローカルモデルディレクトリを明示的に参照し、`~/.paddlex` へのフォールバックや起動時ダウンロードを行わない構成です。

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

## Debug Web

`debug-web` は 1 画面の簡易 UI です。複数画像 upload、RabbitMQ publish、画像ごとの取り込み状態表示、検索ボックス、検索結果の画像 proxy と score breakdown 表示を行います。

```bash
docker compose up --build debug-web gateway rabbitmq seaweedfs qdrant dense-service sparse-service ocr-service caption-service
```

起動後:

- Debug Web: `http://localhost:8080`
- Gateway Search API: `http://localhost:8000/search`

主な環境変数:

- `DEBUG_WEB_HOST`, `DEBUG_WEB_PORT`
- `GATEWAY_BASE_URL`
- `SQLITE_PATH`
- `STATUS_POLL_INTERVAL_MS`
- `SEARCH_TIMEOUT_SECONDS`
- `UPLOAD_OBJECT_PREFIX`
- `RABBITMQ_URL`, `RABBITMQ_QUEUE`, `RABBITMQ_RESULT_QUEUE`
- `S3_ENDPOINT_URL`, `S3_BUCKET`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`
- `AWS_WEB_IDENTITY_TOKEN_FILE`, `AWS_ROLE_ARN`, `AWS_REGION`, `AWS_ENDPOINT_URL_S3`, `AWS_ENDPOINT_URL_STS`

## Runtime Notes

- `image_id` は S3 object key をそのまま使います
- `gateway` は image job を batch 収集し、SeaweedFS から画像を取得して取り込みます
- dense / sparse は batch endpoint でまとめて推論します
- OCR / caption は画像ごとに service を呼びます
- `gateway` の S3 client は従来の `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY` に加えて、`AWS_WEB_IDENTITY_TOKEN_FILE` / `AWS_ROLE_ARN` / `AWS_REGION` / `AWS_ENDPOINT_URL_S3` / `AWS_ENDPOINT_URL_STS` による STS(Web Identity) credential chain に対応します

## Entry Points

- `scripts/run_gateway.py`
- `scripts/run_dense_service.py`
- `scripts/run_sparse_service.py`
- `scripts/run_ocr_service.py`
- `scripts/run_caption_service.py`
