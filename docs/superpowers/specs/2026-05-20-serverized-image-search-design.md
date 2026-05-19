# Serverized Image Search Design

## Summary

既存の `scripts/ingest_images.py` と `scripts/search_images.py` を中心としたスクリプト実行前提の構成を、複数の常駐サービスへ移行する。`gateway` は `HTTP Search API` と `RabbitMQ Consumer` を持ち、画像アップロード後に RabbitMQ に流された `image_id` を契機に SeaweedFS の S3 互換ストレージから画像を取得して取り込む。推論は `dense`, `sparse`, `ocr`, `caption` の各モデルサービスへ HTTP で委譲し、サービスごとに CPU / CUDA をデプロイ時固定で選べるようにする。検索では既存の dense / OCR sparse / Florence sparse を用いた Qdrant 検索を継続しつつ、重みや上限件数をリクエストごとに optional に変更できるようにする。

## Goals

- 常駐サーバとして動作し、手動スクリプト起動なしで画像取り込みと検索を提供する
- `image_id = S3 object key` として RabbitMQ メッセージを処理し、SeaweedFS から対象画像を取得して Qdrant に登録する
- 連続投入された画像を短時間バッファし、可能な範囲で batch 推論する
- 検索を HTTP API で提供し、`ocr_weight`, `dense_weight`, `florence_weight`, `limit` を optional 引数で上書きできるようにする
- モデルごとに CPU / CUDA のどちらで動かすかをデプロイ時に固定できるようにする
- Docker Compose と Dockerfile を用意し、各サービスをコンテナとして実行できるようにする
- ディレクトリ内画像をまとめて S3 に put し、RabbitMQ に publish するテスト用スクリプトと、検索 API を叩くテスト用クライアントを用意する

## Non-Goals

- 画像アップロード用の本番 API はこのアプリに持たせない
- Kubernetes manifest の本格整備までは行わず、まずはコンテナ化と Compose で再現可能な実装を作る
- gRPC の導入は行わない

## Context

現状の構成では、Qdrant のみが `compose.yaml` で起動し、画像取り込みと検索はそれぞれ CLI スクリプトで都度モデルを読み込んで実行する。当初は単一プロセスに全モデルを集約する案が合理的だったが、新要件として「モデルごとに CPU / CUDA をデプロイ時固定で切り替えたい」が追加された。この要件では、`dense は CUDA`, `sparse は CPU` のような非対称配置を許すため、モデル境界でサービスを分ける必要がある。

## Chosen Architecture

### Gateway + Model Services

複数サービス構成を採用する。責務は以下の通り。

- `gateway`
  - `HTTP Search API`
  - `RabbitMQ Consumer`
  - `Ingestion Batch Scheduler`
  - SeaweedFS からの画像取得
  - Qdrant への upsert / search
- `dense-service`
  - `WaonSiglipEncoder`
- `sparse-service`
  - `SpladeJapaneseSparseEncoder`
- `ocr-service`
  - `PaddleOcrVlTextExtractor`
- `caption-service`
  - `Florence2WithJapaneseTranslation`

各モデルサービスは独立コンテナとして動かし、それぞれの Deployment ごとに CPU 版 image / CUDA 版 image または起動設定を選べるようにする。これにより、モデルごとの実行デバイスを個別に固定できる。

### External Dependencies

- `Qdrant`: ベクトル格納と検索
- `RabbitMQ`: 画像取り込みジョブのキュー
- `SeaweedFS S3`: 画像オブジェクトの保存先

### Service Communication

内部通信は HTTP を採用する。主な理由は以下。

- 既存 Python プロジェクトに最短で追加できる
- gRPC よりデバッグが単純
- Compose / Kubernetes の疎通確認が容易
- 今回のボトルネックはモデル推論そのものであり、HTTP のオーバーヘッドが支配的になる可能性は低い

初期実装では `gateway` が画像をダウンロードし、必要に応じて前処理した画像をモデルサービスへ multipart body または一時ファイル経由で渡す。text 系は JSON body を使う。

## Data Flow

### Ingestion

1. 外部スクリプトがローカルディレクトリ内の画像を SeaweedFS S3 に upload する
2. 同スクリプトが `image_id` として S3 object key を RabbitMQ に publish する
3. `gateway` が queue を long-running で監視する
4. Consumer は短い時間窓または件数上限まで `image_id` を貯める
5. `gateway` が SeaweedFS から対象画像を取得し、必要に応じて EXIF 正規化と resize を行う
6. `gateway` が `ocr-service` と `caption-service` を呼び出して各画像の文字情報を得る
7. `gateway` が `dense-service` と `sparse-service` を呼び出して batch 推論する
8. `gateway` が `ImageDocument` を生成して Qdrant へ upsert する
9. 成功分を ack する

### Search

1. クライアントが `gateway` の HTTP API に自然言語クエリを送る
2. `gateway` が `dense-service` と `sparse-service` に query text を送り、query 用ベクトルを生成する
3. `gateway` が Qdrant に dense / OCR sparse / Florence sparse の prefetch 検索を発行する
4. `gateway` が指定または既定の重みで RRF を計算し、結果を返す

## Why HTTP Instead of gRPC

外部公開 API と内部サービス通信の両方で HTTP を採用する。主な理由は以下。

- 既存 Python プロジェクトへの追加実装が最も単純
- テスト用クライアントを最短で用意できる
- Kubernetes 上での疎通確認やデバッグが容易
- 今回の構成では、内部 RPC のレイテンシよりモデル推論時間の方が支配的である可能性が高い

将来的に内部推論サービス間の往復がボトルネックになったら、その時点で gRPC を追加する。

## API Design

### External API: `POST /search`

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
      "image_id": "folder/example.jpg",
      "score": 1.75,
      "path": "s3://images/folder/example.jpg",
      "text": "OCR and caption text"
    }
  ],
  "diagnostics": null
}
```

`include_diagnostics = true` のときは `diagnostics.dense`, `diagnostics.ocr`, `diagnostics.florence` を追加し、現在の CLI `--json` 相当の情報を返す。

### Internal Model APIs

- `dense-service`
  - `POST /encode/image-batch`
  - `POST /encode/text-batch`
- `sparse-service`
  - `POST /encode/text-batch`
- `ocr-service`
  - `POST /extract`
- `caption-service`
  - `POST /caption`

画像系 endpoint は multipart upload を受ける。text 系 endpoint は JSON の文字列配列を受ける。

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
- OCR / caption はまず画像ごとに各サービスを呼び出す
- dense / sparse は batch 対応 endpoint を持たせ、まとめて推論する
- Qdrant upsert は batch 単位でまとめる

この構成により、連続投入時は dense / sparse の効率を上げつつ、低頻度投入時も待ちすぎない。

## Failure Handling

### Retryable

- SeaweedFS 一時的接続失敗
- RabbitMQ 接続断
- Qdrant 一時的接続失敗
- モデルサービス一時的接続失敗

これらは再試行対象で、RabbitMQ では `requeue` または再配信で吸収する。

### Non-Retryable

- object key に対応する画像が存在しない
- 対象ファイルが画像として読めない
- OCR / caption の両方が空で `ExtractedImageText.create()` が失敗する
- モデルサービスが入力エラーを返した

これらは恒久失敗として扱い、再試行しても改善しないため dead-letter に送れる構成を優先する。少なくともログで識別可能にする。

### Partial Failure

batch 内の 1 件が失敗しても全件巻き戻しはしない。可能な限り件別に成功 / 失敗を分離し、成功したものは upsert と ack を進める。

## Domain and Code Structure Changes

### New Responsibilities

- `gateway module`: HTTP API 起動、依存初期化、consumer 起動
- `queue module`: RabbitMQ consume / ack / nack / publish contract
- `storage module`: S3 object download
- `ingestion service`: queue message を画像取り込みワークフローへ変換
- `api schema`: search request / response validation
- `model service modules`: dense / sparse / ocr / caption の個別 HTTP API

### Existing Code Reuse

- `prepare_inference_image()` の前処理は `gateway` 側で再利用する
- `IngestionBatchAccumulator` の batch 化ロジックは `gateway` 向けに移すか共通化する
- `QdrantImageIndex` は `gateway` の upsert と検索基盤として再利用する
- `WaonSiglipEncoder`, `SpladeJapaneseSparseEncoder`, `PaddleOcrVlTextExtractor`, `Florence2WithJapaneseTranslation` は各モデルサービス内部でそのまま使う

### Required Refactors

- `ImageId.from_path()` はローカルファイルパス前提なので、S3 object key ベースの ID を直接扱える生成手段が必要
- `ImagePath.create()` はローカルファイル存在チェック前提なので、S3 object key / 一時ファイル / 論理パスの扱いを分離する必要がある
- 重みを固定定数ではなく、検索時に optional override できるよう `QdrantImageIndex` の検索 API を拡張する
- 各モデルサービス向けに text / image batch endpoint 契約を追加する

## Compose and Containers

`compose.yaml` に少なくとも以下を定義する。

- `qdrant`
- `rabbitmq`
- `seaweedfs-master`
- `seaweedfs-volume`
- `seaweedfs-s3`
- `gateway`
- `dense-service`
- `sparse-service`
- `ocr-service`
- `caption-service`

### Docker Strategy

- サービスごとに CPU / CUDA を選べるようにする
- 単一 Dockerfile を build argument で切り替えられるならそれを優先する
- それが難しければサービスごとに CPU / CUDA image を分ける
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

`gateway` の HTTP API に対して検索リクエストを投げ、結果を簡単に表示する CLI を追加する。JSON 出力オプションも持たせる。

## Testing Strategy

- ドメイン: S3 key ベースの `image_id` / path 取り扱いのテスト
- API: search request validation と optional weight 上書きのテスト
- Queue: message decode と batch flush 条件のテスト
- Storage: object download のインターフェーステスト
- Model service: image / text endpoint の unit test
- Integration: RabbitMQ / SeaweedFS を使わない単体テストを中心にし、外部依存は adapter 境界で fake を使う

## Rollout Notes

- 初期リリースから `gateway` と各モデルサービスを分ける
- Kubernetes ではサービスごとに独立 Deployment を持てる形にする
- 各モデルサービスの CPU / CUDA 選択は Deployment ごとに固定する
