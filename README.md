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

```bash
uv run --group vision python scripts/ingest_images.py ./images --collection picca_images
```

## Search

```bash
uv run python scripts/search_images.py "赤い鳥居が写っている写真" --collection picca_images --limit 5
```

## Architecture

- `src/picca_search/domain.py`: `ImageId`, `ImagePath`, `DenseVector`, `SparseVector`, `SearchQuery` などのドメイン型
- `src/picca_search/application.py`: 取り込みと検索のワークフロー
- `src/picca_search/infrastructure/qdrant_index.py`: Qdrant collection, upsert, Prefetch + RRF search
- `src/picca_search/infrastructure/embedding_models.py`: WAON SigLIP と light-SPLADE の adapter
- `src/picca_search/infrastructure/vision_language_models.py`: PaddleOCR-VL と Florence-2 の adapter
- `scripts/ingest_images.py`: 画像取り込みの composition root
- `scripts/search_images.py`: 検索の composition root

PaddleOCR-VL は公式ドキュメントが推奨する `PaddleOCRVL(pipeline_version="v1")` のパイプラインを使います。Florence-2 は Hugging Face の model card にある `<MORE_DETAILED_CAPTION>` task を使います。
