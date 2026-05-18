# picca-ai-prototype

`docs/architecture.md` に基づく、自然言語で画像を検索するための Python Prototype です。
Qdrant は Docker Compose で動かし、画像取り込みと検索は単純な Python スクリプトとして実行します。

## Setup

```bash
uv python pin 3.13
uv sync --group vision
docker compose up -d qdrant
```

## Ingest

画像ディレクトリ内の `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif` を Qdrant に登録します。
Sparse Vector 用テキストは、PaddleOCR-VL による OCR と Florence-2 による Caption を統合して作ります。
Florence-2 は英語でキャプションを生成するため、Helsinki-NLP/opus-mt-en-jap で日本語に翻訳してから Sparse Encoder に入れます。
推論前には EXIF Orientation を正規化します。長辺が 2048px を超える画像は ingest 時に推論用の一時ファイルへリサイズされます。透過画像と `.png` / `.bmp` は一時ファイルでも PNG を維持し、それ以外の非透過画像は JPEG に再エンコードします。元の画像ファイルと保存される画像パスはそのまま維持されます。

```bash
uv run --group vision python scripts/ingest_images.py ./images --collection picca_images
```

各モデルは `--dense-device`, `--sparse-device`, `--caption-device`, `--translator-device`, `--ocr-device` で
`cuda` / `mps` / `cpu` を個別指定できます。未指定時は各モデルの自動選択を使います。
Qdrant への登録は `--batch-size` 件ごとにまとめて upsert します。既定値は `16` です。

## Search

```bash
uv run python scripts/search_images.py "赤い鳥居が写っている写真" --collection picca_images --limit 5
```

検索時も `--dense-device`, `--sparse-device` で個別指定できます。未指定時は自動選択です。
通常検索は Qdrant の fusion query を使う高速経路を通ります。`--explain` と `--json` は
dense / sparse の個別順位を出すため、診断用の追加クエリを実行します。

## Architecture

- `src/picca_search/domain.py`: `ImageId`, `ImagePath`, `DenseVector`, `SparseVector`, `SearchQuery` などのドメイン型
- `src/picca_search/application.py`: 取り込みと検索のワークフロー
- `src/picca_search/infrastructure/qdrant_index.py`: Qdrant collection, upsert, Prefetch + RRF search
- `src/picca_search/infrastructure/embedding_models.py`: WAON SigLIP と light-SPLADE の adapter
- `src/picca_search/infrastructure/vision_language_models.py`: PaddleOCR-VL、Florence-2、MarianMT 翻訳の adapter
- `scripts/ingest_images.py`: 画像取り込みの composition root
- `scripts/search_images.py`: 検索の composition root

PaddleOCR-VL は公式ドキュメントが推奨する `PaddleOCRVL(pipeline_version="v1")` のパイプラインを使います。Florence-2 は Hugging Face の model card にある `<MORE_DETAILED_CAPTION>` task を使います。
