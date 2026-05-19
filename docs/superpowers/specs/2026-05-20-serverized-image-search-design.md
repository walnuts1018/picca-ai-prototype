# Serverized Image Search Design

## Summary

既存の `scripts/ingest_images.py` と `scripts/search_images.py` を中心としたスクリプト実行前提の構成を、単一の常駐アプリケーションへ移行する。常駐アプリは `HTTP Search API` と `RabbitMQ Consumer` を同一プロセスで持ち、画像アップロード後に RabbitMQ に流された `image_id` を契機に SeaweedFS の S3 互換ストレージから画像を取得して取り込む。検索では既存の dense / OCR sparse / Florence sparse を用いた Qdrant 検索を継続しつつ、重みや上限件数をリクエストごとに optional に変更できるようにする。

## Goals

- 常駐サーバとして動作し、手動スクリプト起動なしで画像取り込みと検索を提供する
- `image_id = S3 object key` として RabbitMQ メッセージを処理し、SeaweedFS から対象画像を取得して Qdrant に登録する
- 連続投入された画像を短時間バッファし、可能な範囲で batch 推論する
- 検索を HTTP API で提供し、`ocr_weight`, `dense_weight`, `florence_weight`, `limit` を optional 引数で上書きできるようにする
- Docker Compose と Dockerfile を用意し、CPU / CUDA の両方でコンテナ実行できるようにする
- ディレクトリ内画像をまとめて S3 に put し、RabbitMQ に publish するテスト用スクリプトと、検索 API を叩くテスト用クライアントを用意する

## Non-Goals

- 画像アップロード用の本番 API はこのアプリに持たせない
- この段階では OCR / Dense / Sparse / Caption を独立マイクロサービスに分割しない
- Kubernetes manifest の本格整備までは行わず、まずはコンテナ化と Compose で再現可能な実装を作る

## Context

現状の構成では、Qdrant のみが `compose.yaml` で起動し、画像取り込みと検索はそれぞれ CLI スクリプトで都度モデルを読み込んで実行する。これを単純に `API Server` と `Worker` に分割すると、検索側でも `WaonSiglipEncoder` と `SpladeJapaneseSparseEncoder` を使うため、dense / sparse モデルが二重ロードされる。GPU メモリや起動時間を考えると、この段階ではモデルを 1 プロセスに集約し、検索と取り込みの両方で共有するほうが合理的である。

## Chosen Architecture

### Single Inference Application

単一の常駐アプリケーションを採用する。1 プロセス内で以下を同時に動かす。

- `HTTP Search API`
- `RabbitMQ Consumer`
- `Ingestion Batch Scheduler`
- 推論モデル群
  - `WaonSiglipEncoder`
  - `SpladeJapaneseSparseEncoder`
  - `PaddleOcrVlTextExtractor`
  - `Florence2WithJapaneseTranslation`

この構成により、検索と取り込みの両方が同じモデルインスタンスを共有できる。将来、検索レイテンシと取り込みスループットの競合が実測で問題になった場合に限り、モデルサーバ分離を検討する。

### External Dependencies

- `Qdrant`: ベクトル格納と検索
- `RabbitMQ`: 画像取り込みジョブのキュー
- `SeaweedFS S3`: 画像オブジェクトの保存先

## Data Flow

### Ingestion

1. 外部スクリプトがローカルディレクトリ内の画像を SeaweedFS S3 に upload する
2. 同スクリプトが `image_id` として S3 object key を RabbitMQ に publish する
3. 常駐アプリが queue を long-running で監視する
4. Consumer は短い時間窓または件数上限まで `image_id` を貯める
5. SeaweedFS から対象画像を取得し、必要に応じて EXIF 正規化と resize を行う
6. 各画像について OCR と caption を生成する
7. dense / sparse は既存の batch エンコードパスを使ってまとめて推論する
8. `ImageDocument` を生成して Qdrant へ upsert する
9. 成功分を ack する

### Search

1. クライアントが HTTP API に自然言語クエリを送る
2. 常駐アプリが query text から dense vector と sparse vector を生成する
3. Qdrant に dense / OCR sparse / Florence sparse の prefetch 検索を発行する
4. 指定または既定の重みで RRF を計算し、結果を返す

## Why HTTP Instead of gRPC

HTTP API を採用する。主な理由は以下。

- 既存 Python プロジェクトへの追加実装が最も単純
- テスト用クライアントを最短で用意できる
- Kubernetes 上での疎通確認やデバッグが容易
- 今回のリクエスト / レスポンス規模では gRPC の性能差より実装単純性の恩恵が大きい

将来的に cross-service 推論や高頻度な内部 RPC が必要になったら、その時点で gRPC を追加する。

## API Design

### `POST /search`

Request body:

```json
{
  "query": "赤い鳥居が写っている写真",
  "limit": 10,
  "dense_weight": 4.0,
  "ocr_weight": 2.0,
  "florence_weight": 1.0,
  "include_diagnostics": false
}
```

Rules:

- `query` は必須、空白のみは不可
- `limit` は optional、未指定時は既定値を使う
- `dense_weight`, `ocr_weight`, `florence_weight` は optional、未指定時は既存デフォルト値を使う
- `include_diagnostics` は optional、`true` のとき dense / ocr / florence の個別順位を返す

Response body:

```json
{
  "query": "赤い鳥居が写っている写真",
  "limit": 10,
  "weights": {
    "dense": 4.0,
    "ocr": 2.0,
    "florence": 1.0
  },
  "results": [
    {
      "image_id": "bucket/path/example.jpg",
      "score": 1.75,
      "path": "s3://images/example.jpg",
      "text": "OCR and caption text"
    }
  ],
  "diagnostics": null
}
```

`include_diagnostics = true` のときは `diagnostics.dense`, `diagnostics.ocr`, `diagnostics.florence` を追加し、現在の CLI `--json` 相当の情報を返す。

## Queue Message Contract

RabbitMQ のメッセージ本体は最小限の JSON とする。

```json
{
  "image_id": "folder/example.jpg"
}
```

ここで `image_id` はそのまま S3 object key として扱う。画像メタデータを増やす余地は残すが、この時点では YAGNI を優先して追加しない。

## Batch Ingestion Strategy

- `max_batch_size`: 件数上限で flush
- `max_batch_wait_ms`: 最初の 1 件を受け取ってからの待機上限で flush
- batch 対象は dense / sparse のエンコードを中心にする
- OCR / caption はモデルの事情上、まずは画像ごとに処理してよい
- Qdrant upsert は batch 単位でまとめる

この構成により、連続投入時は dense / sparse の効率を上げつつ、低頻度投入時も待ちすぎない。

## Failure Handling

### Retryable

- SeaweedFS 一時的接続失敗
- RabbitMQ 接続断
- Qdrant 一時的接続失敗

これらは再試行対象で、RabbitMQ では `requeue` または再配信で吸収する。

### Non-Retryable

- object key に対応する画像が存在しない
- 対象ファイルが画像として読めない
- OCR / caption の両方が空で `ExtractedImageText.create()` が失敗する

これらは恒久失敗として扱い、再試行しても改善しないため dead-letter に送れる構成を優先する。少なくともログで識別可能にする。

### Partial Failure

batch 内の 1 件が失敗しても全件巻き戻しはしない。可能な限り件別に成功 / 失敗を分離し、成功したものは upsert と ack を進める。

## Domain and Code Structure Changes

### New Responsibilities

- `server module`: HTTP API 起動、依存初期化、consumer 起動
- `queue module`: RabbitMQ consume / ack / nack / publish contract
- `storage module`: S3 object download
- `ingestion service`: queue message を画像取り込みワークフローへ変換
- `api schema`: search request / response validation

### Existing Code Reuse

- `prepare_inference_image()` の前処理は再利用する
- `IngestionBatchAccumulator` の batch 化ロジックは server 向けに移すか共通化する
- `QdrantImageIndex` は upsert と検索基盤として再利用する
- `WaonSiglipEncoder`, `SpladeJapaneseSparseEncoder`, `PaddleOcrVlTextExtractor`, `Florence2WithJapaneseTranslation` はそのまま使う

### Required Refactors

- `ImageId.from_path()` はローカルファイルパス前提なので、S3 object key ベースの ID を直接扱える生成手段が必要
- `ImagePath.create()` はローカルファイル存在チェック前提なので、S3 object key / 一時ファイル / 論理パスの扱いを分離する必要がある
- 重みを固定定数ではなく、検索時に optional override できるよう `QdrantImageIndex` の検索 API を拡張する

## Compose and Containers

`compose.yaml` に少なくとも以下を定義する。

- `qdrant`
- `rabbitmq`
- `seaweedfs-master`
- `seaweedfs-volume`
- `seaweedfs-s3`
- `app`

### Docker Strategy

- 可能なら単一 Dockerfile で `cpu` / `cuda` を build argument で切り替える
- それが煩雑なら CPU / CUDA を分ける
- CUDA 版は `nvidia/cuda:*-cudnn-devel-ubuntu24.04` や `nvidia/cuda:*-base-ubuntu24.04` を土台にする
- `uv` を使った依存解決を維持する
- multi-stage build と cache mount を使って依存解決と wheel install を高速化する

## Test Utilities

### Bulk Upload Script

ローカルディレクトリを走査して:

1. SeaweedFS S3 に画像を upload
2. object key を `image_id` として RabbitMQ に publish

これにより end-to-end テスト投入が可能になる。

### Search Test Client

HTTP API に対して検索リクエストを投げ、結果を簡単に表示する CLI を追加する。JSON 出力オプションも持たせる。

## Testing Strategy

- ドメイン: S3 key ベースの `image_id` / path 取り扱いのテスト
- API: search request validation と optional weight 上書きのテスト
- Queue: message decode と batch flush 条件のテスト
- Storage: object download のインターフェーステスト
- Integration: RabbitMQ / SeaweedFS を使わない単体テストを中心にし、外部依存は adapter 境界で fake を使う

## Rollout Notes

- 初期リリースでは単一コンテナに全モデルを載せる
- Kubernetes ではまず 1 Deployment として運用する
- 検索レイテンシ悪化や GPU 競合が実測で顕著な場合にのみ、モデルサーバ分離を再評価する
