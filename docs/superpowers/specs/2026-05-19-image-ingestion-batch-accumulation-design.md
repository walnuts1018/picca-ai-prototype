# Design: 画像取り込みの非同期バッチ化（バッチアキュムレータ方式）

## 目的

画像取り込み時のモデル推論（SigLIP Dense Encoder、SPLADE Sparse Encoder）をバッチ化し、GPU効率とスループットを向上させる。OCRとFlorence-2のキャプション生成は引き続き逐次実行する。

## アーキテクチャ

### データフロー

```
画像1 → OCR/Caption（逐次） → バッファに追加
画像2 → OCR/Caption（逐次） → バッファに追加
...
画像N → OCR/Caption（逐次） → バッファに追加
         → バッファがバッチサイズに到達
         → バッチエンコード（Dense + Sparse）
         → ImageDocument 生成
         → Qdrant upsert
```

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `embedding_models.py` | `encode_images()` (WaonSiglipEncoder), `encode_texts()` (SpladeJapaneseSparseEncoder) バッチメソッド追加 |
| `ingest_images.py` | `IngestionBatchAccumulator` クラス追加、`ingest_images()` 書き換え |
| その他 | 変更なし |

## 詳細設計

### 1. WaonSiglipEncoder.encode_images()

```python
def encode_images(self, images: list[Image.Image]) -> list[DenseVector]:
    inputs = self.processor(images=images, return_tensors="pt").to(self.device)
    with self.torch.no_grad():
        features = self.model.get_image_features(**inputs)
    # features.shape = (batch_size, hidden_dim)
    normalized = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return [DenseVector.create(row.tolist()) for row in normalized]
```

- 既存の `encode_image()` は残す（後方互換性）
- 呼び出し元で `Image.open().convert("RGB")` 済みのオブジェクトを渡す
- `_normalized_values` ヘルパーを2Dテンソル対応にリファクタリングし、単一ソースオブ真理とする:

```python
def _normalized_values(torch, tensor) -> list[float]:
    # 1D (single vector) or 2D (batch) 両対応
    if tensor.dim() == 1:
        normalized = tensor / tensor.norm().clamp(min=1e-12)
        return normalized.detach().cpu().tolist()
    normalized = tensor / tensor.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return normalized.detach().cpu().tolist()
```

### 2. SpladeJapaneseSparseEncoder.encode_texts()

```python
def encode_texts(self, texts: list[str]) -> list[SparseVector]:
    inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
    with self.torch.no_grad():
        logits = self.model(**inputs).logits
        weights = self.torch.log1p(self.torch.relu(logits))
        weights = weights * inputs["attention_mask"].unsqueeze(-1)
        pooled = self.torch.max(weights, dim=1).values
    # 各テキストごとに top_k 処理
    results = []
    for i in range(len(texts)):
        vector = pooled[i]
        values, indices = self.torch.topk(vector, k=min(self.top_k, vector.shape[0]))
        non_zero = values > 0
        values = values[non_zero].detach().cpu().tolist()
        indices = indices[non_zero].detach().cpu().tolist()
        if len(indices) == 0:
            unknown_token_id = self.tokenizer.unk_token_id or 0
            results.append(SparseVector.create([unknown_token_id], [1.0]))
        else:
            results.append(SparseVector.create(indices, values))
    return results
```

- 既存の `encode_text()` は残す
- `padding=True` でバッチ内の最大長に揃える

### 3. IngestionBatchAccumulator

```python
@dataclass
class _PendingImage:
    image_path: Path
    image: Image.Image  # RGB変換済み
    ocr_text: str
    caption: str

class IngestionBatchAccumulator:
    # 2048px RGB画像は約12MB。batch_size=128で約1.5GBのRAMを消費するため上限を設ける
    MAX_BATCH_SIZE = 64

    def __init__(
        self,
        *,
        image_dense_encoder: WaonSiglipEncoder,
        sparse_encoder: SpladeJapaneseSparseEncoder,
        batch_size: int,
    ):
        if batch_size > self.MAX_BATCH_SIZE:
            raise ValueError(
                f"batch_size={batch_size} exceeds MAX_BATCH_SIZE={self.MAX_BATCH_SIZE}. "
                f"Large batches can cause OOM due to in-memory PIL.Image storage."
            )
        self.encoder = image_dense_encoder
        self.sparse = sparse_encoder
        self.batch_size = batch_size
        self.pending: list[_PendingImage] = []

    def add(self, image_path: Path, image: Image.Image, ocr_text: str, caption: str) -> None:
        # バリデーションを早期実行し、バッチ全体が失敗するのを防ぐ
        ExtractedImageText.create(ocr_text, caption)
        self.pending.append(_PendingImage(image_path, image, ocr_text, caption))

    def is_ready(self) -> bool:
        return len(self.pending) >= self.batch_size

    def flush(self) -> list[ImageDocument]:
        if not self.pending:
            return []
        pending = self.pending
        self.pending = []
        # バッチエンコード
        images = [p.image for p in pending]
        texts = [ExtractedImageText.create(p.ocr_text, p.caption).combined for p in pending]
        dense_vectors = self.encoder.encode_images(images)
        sparse_vectors = self.sparse.encode_texts(texts)
        # ドキュメント生成（バリデーション済みなので失敗しない）
        documents = []
        for p, dense, sparse, text in zip(pending, dense_vectors, sparse_vectors, texts):
            doc = ImageDocument.create(
                image_id=ImageId.from_path(p.image_path),
                image_path=ImagePath.create(p.image_path),
                dense_vector=dense,
                sparse_vector=sparse,
                text=text,
                ocr_text=p.ocr_text,
                caption=p.caption,
            )
            documents.append(doc)
        return documents
```

### 4. ingest_images() の書き換え

`prepare_inference_image` はテンポラリファイルを生成する可能性があるため、コンテキスト終了後にファイルが消えるとバッチエンコードが失敗する。
これを避けるため、`IngestionBatchAccumulator` は画像パスの代わりに `PIL.Image` オブジェクトを保持する。

```python
@dataclass
class _PendingImage:
    image_path: Path
    image: Image.Image  # メモリ上に保持
    ocr_text: str
    caption: str
```

```python
def ingest_images(
    *,
    image_paths: list[Path],
    ocr_text_extractor: PaddleOcrVlTextExtractor,
    image_captioner: Florence2Captioner,
    image_dense_encoder: WaonSiglipEncoder,
    sparse_encoder: SpladeJapaneseSparseEncoder,
    image_index: QdrantImageIndex,
    batch_size: int,
) -> list[ImageDocument]:
    accumulator = IngestionBatchAccumulator(
        image_dense_encoder=image_dense_encoder,
        sparse_encoder=sparse_encoder,
        batch_size=batch_size,
    )
    documents: list[ImageDocument] = []

    for image_path in image_paths:
        with prepare_inference_image(image_path) as inference_path:
            ocr_text = ocr_text_extractor.extract_text(inference_path)
            caption = image_captioner.caption(inference_path)
            with Image.open(inference_path).convert("RGB") as img:
                accumulator.add(image_path, img.copy(), ocr_text, caption)

        if accumulator.is_ready():
            batch_docs = accumulator.flush()
            image_index.upsert(batch_docs)
            documents.extend(batch_docs)

    # 残り
    remaining = accumulator.flush()
    if remaining:
        image_index.upsert(remaining)
        documents.extend(remaining)

    return documents
```

`IngestionBatchAccumulator.flush()` では、保持している `Image` オブジェクトを直接 `processor(images=...)` に渡す。

## エラーハンドリング

- `accumulator.add()` で `ExtractedImageText.create()` を早期実行し、OCR/Captionが両方空の場合に即座に `ValueError` を発生させる（バッチ全体の失敗を防ぐ）
- バッチエンコード中に例外が発生した場合、そのバッチ全体が失敗する（既存と同じ）
- 個々の画像の失敗をスキップする仕組みはスコープ外（必要であれば別イシュー）
- `batch_size` が `MAX_BATCH_SIZE=64` を超える場合、`ValueError` を発生（メモリ圧迫の防止）

## テスト計画

- `WaonSiglipEncoder.encode_images()`: 複数画像のバッチエンコードが単一実行と同等の結果を返す
- `SpladeJapaneseSparseEncoder.encode_texts()`: 複数テキストのバッチエンコードが単一実行と同等の結果を返す
- `IngestionBatchAccumulator`: バッチサイズでflushされること、残りが正しく処理されること
- `IngestionBatchAccumulator.add()`: OCR/Captionが両方空の場合に即座に `ValueError` を発生
- `IngestionBatchAccumulator`: `batch_size > MAX_BATCH_SIZE` で `ValueError` を発生
- 既存の `ingest_images` CLIが従来通り動作すること
