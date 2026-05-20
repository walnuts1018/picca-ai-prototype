# Debug Web Application Design

**Date:** 2026-05-20

## Goal

既存の `gateway + model services + RabbitMQ + SeaweedFS + Qdrant` 構成に対して、debug 用の簡単な Web アプリケーションを追加する。  
この Web アプリケーションは 1 画面で以下を行えることを目的とする。

- 複数画像の S3 へのアップロード
- RabbitMQ への image job message 配信
- 画像ごとの ingestion 状態表示
- 文字列検索
- 検索結果の画像、総合スコア、各要素スコア、caption の表示

本番用 UI ではなく debug 用 UI とし、見た目は素朴でよい。  
ただしコンテナとしてデプロイ可能であり、設定は環境変数から行えるようにする。

## Scope

今回の対象は以下とする。

- Go 製 `debug-web` サービスの新設
- `gateway` の検索 API 拡張
- `gateway` からの ingestion 結果 event 配信
- `compose.yaml` と Dockerfile の追加更新
- unit test の追加

今回の対象外は以下とする。

- e2e テスト
- 認証機構の追加
- 本番公開向け UI/UX 改善
- WebSocket や SSE によるリアルタイム push
- `gateway` に新たな永続ストレージを持たせること

## Chosen Approach

実装方式は `Go 単体の Debug Web + Gateway API 拡張` とする。

この方針を選ぶ理由は以下。

- debug 用 UI に対して React/TanStack Start は過剰
- Go 単体なら単一バイナリ・単一コンテナで配備が簡単
- 既存の Python/Gateway 側には必要最小限の API 拡張だけを入れられる
- ingestion 状態の永続化を `debug-web` 側 sqlite に閉じ込められる

## High-Level Architecture

構成要素は以下。

- `debug-web`
  - 1 画面 HTML を返す Go HTTP サーバ
  - S3 upload
  - RabbitMQ への image job publish
  - RabbitMQ からの result event consume
  - sqlite による画像状態管理
  - S3 object の image proxy
- `gateway`
  - 既存の検索 API
  - 既存の RabbitMQ consumer
  - 検索結果 breakdown 付きレスポンス
  - ingestion 結果 event の publish
- `rabbitmq`
  - `image_jobs` queue
  - `image_job_results` queue
- `seaweedfs`
  - 元画像保管
- `qdrant`
  - vector index

## Data Flow

### Upload / Ingestion

1. ユーザーが `debug-web` に複数画像を multipart upload
2. `debug-web` が object key を生成して S3 に upload
3. `debug-web` が sqlite に画像レコードを `queued` として保存
4. `debug-web` が `image_jobs` queue に画像ごとの message を publish
5. `gateway` が queue から message を取得
6. `gateway` が画像取得、OCR、caption、embedding、Qdrant upsert を実行
7. `gateway` が画像ごとの結果 event を `image_job_results` queue に publish
8. `debug-web` の background consumer が result event を受信し、sqlite の状態を更新
9. ユーザーが画面再読込または polling により `queued / processing / indexed / failed` を確認

### Search

1. ユーザーが `debug-web` の検索フォームに文字列を入力
2. `debug-web` が `gateway /search` を `include_diagnostics=true` で呼ぶ
3. `gateway` が Qdrant と diagnostics 情報から UI 向け breakdown を構築
4. `debug-web` が結果一覧を表示
5. 画像サムネイルは `debug-web /images/{image_id}` 経由で表示

## State Ownership

ingestion 状態の正本は `debug-web` の sqlite とする。  
`gateway` は状態を永続化しない。  
`gateway` は結果 event を publish するだけに留める。

これにより以下を満たす。

- `gateway` に新たな外部データストア依存を追加しない
- `debug-web` 再起動後も状態一覧を保持できる
- 将来的に結果 event を別用途でも再利用できる

## Image Status Model

画像単位の状態を持つ。

- `queued`
  - `debug-web` が S3 upload と job publish を完了した状態
- `processing`
  - `gateway` が画像の処理開始を通知した状態
- `indexed`
  - `gateway` が Qdrant upsert 成功を通知した状態
- `failed`
  - `gateway` がその画像の処理失敗を通知した状態

`debug-web` は結果 event の `occurred_at` を見て最新状態で上書きする。  
同一画像に複数回 `processing` event が来ても問題ない。

## RabbitMQ Message Contracts

### Existing Job Queue

queue 名:

- `image_jobs`

message payload:

```json
{
  "image_id": "debug/2026/05/20/abc123-sample.jpg"
}
```

### New Result Queue

queue 名:

- `image_job_results`

message payload:

```json
{
  "image_id": "debug/2026/05/20/abc123-sample.jpg",
  "status": "indexed",
  "occurred_at": "2026-05-20T12:34:56Z",
  "error_message": null
}
```

`status` は `processing | indexed | failed` を取る。  
`error_message` は `failed` 時だけ入る。

## Gateway API Design

### Search API

既存の `POST /search` を拡張する。  
既存互換は維持しつつ、`include_diagnostics=true` 時のレスポンスに画像単位 breakdown を追加する。

追加レスポンス要素の例:

```json
{
  "query": "赤い鳥居",
  "limit": 10,
  "weights": {
    "dense": 4.0,
    "ocr": 2.0,
    "florence": 1.0
  },
  "results": [
    {
      "image_id": "debug/example.jpg",
      "score": 3.5,
      "path": "s3://images/debug/example.jpg",
      "text": "OCR と caption を結合したテキスト",
      "ocr_text": "神社",
      "caption": "赤い鳥居のある参道",
      "score_breakdown": {
        "dense_score": 2.0,
        "ocr_score": 1.0,
        "florence_score": 0.5,
        "dense_rank": 1,
        "ocr_rank": 1,
        "florence_rank": 1
      }
    }
  ]
}
```

`score_breakdown` は diagnostics の各ランキングを `image_id` で付き合わせて組み立てる。  
計算は現在の `search_with_diagnostics()` と同じく `weight / (k + rank)` を用いる。  
通常検索の総合スコアは引き続き Qdrant 側が RRF で計算する。

### Ingestion Result Publication

`gateway` は API を新設して状態を返すのではなく、まずは result queue に event を publish する。  
状態確認 API は `debug-web` 側 sqlite を見る方が責務分離に合うため、`gateway` には追加しない。

consumer 処理で event を出すタイミング:

- バッチに取り込んで個々の画像処理を始める前後で `processing`
- その画像が upsert 成功したら `indexed`
- その画像が失敗して dead-letter 相当になったら `failed`

batch 単位ではなく画像単位で publish する。

## Debug Web HTTP Design

`debug-web` は単一の Go HTTP サーバとする。

エンドポイント:

- `GET /`
  - upload フォーム、状態一覧、検索フォーム、検索結果を含む 1 画面 HTML
- `POST /uploads`
  - 複数画像 upload、S3 保存、sqlite 記録、RabbitMQ publish
- `GET /uploads/statuses`
  - sqlite から状態一覧を JSON 返却
- `POST /search`
  - `gateway /search` を呼んで検索結果を HTML に再描画
- `GET /images/{image_id}`
  - S3 から object を取得して proxy
- `GET /healthz`
  - liveness 用

background 処理:

- `image_job_results` queue consumer を goroutine で 1 本起動
- event を受けたら sqlite を更新

UI 更新方式:

- debug 用のため polling ベース
- JavaScript は最小限でもよい
- WebSocket/SSE は採用しない

## Debug Web Screen Design

1 画面に以下を配置する。

### Upload Area

- 複数ファイル選択 input
- upload 実行ボタン
- 画像状態テーブル

状態テーブル列:

- thumbnail
- filename
- image_id
- status
- updated_at
- error_message

### Search Area

- query input
- limit input
- search 実行ボタン

検索結果一覧列:

- thumbnail
- image_id
- total_score
- dense_score
- ocr_score
- florence_score
- caption
- ocr_text

画像表示はすべて `debug-web` の proxy URL を使う。

## SQLite Design

`debug-web` は sqlite をローカル永続化として使う。  
最低限のテーブルは 1 つでよい。

想定カラム:

- `image_id` TEXT PRIMARY KEY
- `object_key` TEXT NOT NULL
- `filename` TEXT NOT NULL
- `status` TEXT NOT NULL
- `error_message` TEXT NULL
- `uploaded_at` TEXT NOT NULL
- `last_event_at` TEXT NOT NULL
- `attempt_count` INTEGER NOT NULL DEFAULT 0

必要なら将来のために `created_at` / `updated_at` を分けてもよいが、debug 用の最小実装では `uploaded_at` と `last_event_at` で十分。

## Object Key Strategy

object key は upload ごとに衝突しない形にする。  
例:

```text
debug/2026/05/20/<uuid>-<original-filename>
```

これにより以下を満たす。

- 画像単位の一意性
- 元ファイル名の可読性
- `image_id = object key` という既存 runtime note との整合

## Gateway Implementation Boundary

`gateway` で行う変更は以下に限定する。

- result event message 型の追加
- RabbitMQ への result publish 機能追加
- ingestion 処理から画像単位 event を publish
- `/search` の breakdown 付きレスポンス整形
- 追加環境変数の導入

`gateway` では以下を行わない。

- sqlite 参照
- 追加の永続ストレージ導入
- debug UI の同居

## Debug Web Implementation Boundary

`debug-web` で行う変更は以下。

- Go module の追加
- HTTP server と HTML template
- sqlite repository
- S3 client
- RabbitMQ publisher/consumer
- gateway search client
- image proxy handler
- Dockerfile 追加

## Configuration

### Debug Web

最低限以下の環境変数を持たせる。

- `DEBUG_WEB_HOST`
- `DEBUG_WEB_PORT`
- `RABBITMQ_URL`
- `RABBITMQ_JOB_QUEUE`
- `RABBITMQ_RESULT_QUEUE`
- `S3_ENDPOINT_URL`
- `S3_BUCKET`
- `S3_ACCESS_KEY_ID`
- `S3_SECRET_ACCESS_KEY`
- `AWS_WEB_IDENTITY_TOKEN_FILE`
- `AWS_ROLE_ARN`
- `AWS_REGION`
- `AWS_ENDPOINT_URL_S3`
- `AWS_ENDPOINT_URL_STS`
- `GATEWAY_BASE_URL`
- `SQLITE_PATH`
- `UPLOAD_OBJECT_PREFIX`
- `STATUS_POLL_INTERVAL_MS`
- `SEARCH_TIMEOUT_SECONDS`

S3 credential は `gateway` と同等以上に、静的 credential と Web Identity credential chain の両方に対応する。

### Gateway

追加する環境変数:

- `RABBITMQ_RESULT_QUEUE`

必要なら heartbeat や publish 周りの既存設定を流用する。

## Containerization

### Debug Web Container

- `debug-web.Dockerfile` を新設
- Go の multi-stage build
- 最終 image は単一バイナリを含む軽量 image

### Compose

`compose.yaml` に `debug-web` service を追加する。

依存:

- `rabbitmq`
- `seaweedfs`
- `gateway`

volume:

- sqlite 永続化用 directory

公開ポート:

- `debug-web` の HTTP port

## Testing Strategy

e2e テストは行わない。  
unit test を中心に実装する。

### Go Unit Tests

- sqlite repository
- upload handler
- image proxy handler
- gateway search client
- result queue consumer

### Python Unit Tests

- gateway の result event publish
- ingestion outcome ごとの event 生成
- search breakdown 整形

重点的に確認するポイント:

- batch 処理で画像ごとの success/failure event が正しく分離されること
- breakdown 計算が diagnostics の rank/weight と一致すること
- `debug-web` が result event を受けて sqlite 状態を正しく遷移させること

## Risks and Mitigations

### Risk 1: Batch 処理で画像単位イベントがずれる

Mitigation:

- ingestion 処理で job と結果の対応を明示的に保持する
- unit test で success/failure 混在ケースを検証する

### Risk 2: diagnostics と UI 表示の score 解釈がずれる

Mitigation:

- breakdown は `search_with_diagnostics()` の rank に基づいて一貫計算する
- score 計算 helper を共通化して unit test する

### Risk 3: sqlite と result queue consumer の競合

Mitigation:

- sqlite repository に単純な upsert/update API を用意する
- write path を repository に閉じ込める

## Acceptance Criteria

- `debug-web` から複数画像を upload できる
- upload 後、各画像が `queued / processing / indexed / failed` のいずれかで表示される
- `debug-web` が画像を S3 から proxy 配信できる
- 検索ボックスから文字列検索できる
- 検索結果に画像、総合スコア、dense/ocr/florence の各スコア、caption、ocr_text が表示される
- `debug-web` はコンテナとして起動できる
- `debug-web` の設定は環境変数から行える
- `gateway` は追加ストレージなしで結果 event を publish できる
