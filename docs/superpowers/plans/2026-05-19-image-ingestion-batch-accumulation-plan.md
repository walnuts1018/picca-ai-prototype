# Image Ingestion Batch Accumulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Batch image encoding (SigLIP) and text encoding (SPLADE) during ingestion to improve GPU throughput.

**Architecture:** Add batch methods to encoders, introduce an IngestionBatchAccumulator that buffers OCR/caption results and flushes them as batches for encoding.

**Tech Stack:** Python, PyTorch, transformers, PIL, pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/picca_search/infrastructure/embedding_models.py` | Modify | Add `encode_images()` and `encode_texts()` batch methods; refactor `_normalized_values` |
| `scripts/ingest_images.py` | Modify | Add `_PendingImage` dataclass, `IngestionBatchAccumulator` class; rewrite `ingest_images()` |
| `tests/test_scripts.py` | Modify | Update FakeDenseEncoder/FakeSparseEncoder to support batch methods |

---

### Task 1: Add batch encode methods to WaonSiglipEncoder

**Files:**
- Modify: `src/picca_search/infrastructure/embedding_models.py`
- Test: `tests/test_scripts.py` (existing FakeDenseEncoder will need batch method for compatibility)

- [ ] **Step 1: Refactor `_normalized_values` to handle 1D and 2D tensors**

In `src/picca_search/infrastructure/embedding_models.py`, replace the existing `_normalized_values` function:

```python
def _normalized_values(torch, tensor) -> list[float]:
    if tensor.dim() == 1:
        normalized = tensor / tensor.norm().clamp(min=1e-12)
        return normalized.detach().cpu().tolist()
    normalized = tensor / tensor.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return normalized.detach().cpu().tolist()
```

- [ ] **Step 2: Add `encode_images()` method to `WaonSiglipEncoder`**

Add this method to the `WaonSiglipEncoder` class (after `encode_text`):

```python
    def encode_images(self, images: list[Image.Image]) -> list[DenseVector]:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            features = self.model.get_image_features(**inputs)
        normalized = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return [DenseVector.create(row.tolist()) for row in normalized]
```

- [ ] **Step 3: Run existing tests to verify no regression**

Run: `uv run pytest tests/ -v`
Expected: All existing tests pass

- [ ] **Step 4: Commit**

```bash
git add src/picca_search/infrastructure/embedding_models.py
git commit -m "feat: add batch encode_images to WaonSiglipEncoder and refactor _normalized_values"
```

---

### Task 2: Add batch encode_texts to SpladeJapaneseSparseEncoder

**Files:**
- Modify: `src/picca_search/infrastructure/embedding_models.py`

- [ ] **Step 1: Add `encode_texts()` method to `SpladeJapaneseSparseEncoder`**

Add this method to the `SpladeJapaneseSparseEncoder` class (after `encode_text`):

```python
    def encode_texts(self, texts: list[str]) -> list[SparseVector]:
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            weights = self.torch.log1p(self.torch.relu(logits))
            weights = weights * inputs["attention_mask"].unsqueeze(-1)
            pooled = self.torch.max(weights, dim=1).values
        results: list[SparseVector] = []
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

- [ ] **Step 2: Run existing tests to verify no regression**

Run: `uv run pytest tests/ -v`
Expected: All existing tests pass

- [ ] **Step 3: Commit**

```bash
git add src/picca_search/infrastructure/embedding_models.py
git commit -m "feat: add batch encode_texts to SpladeJapaneseSparseEncoder"
```

---

### Task 3: Add IngestionBatchAccumulator to ingest_images.py

**Files:**
- Modify: `scripts/ingest_images.py`

- [ ] **Step 1: Add imports and `_PendingImage` dataclass**

Add to the imports section in `scripts/ingest_images.py`:

```python
from dataclasses import dataclass

from PIL import Image

from picca_search.domain import ExtractedImageText, ImageDocument, ImageId, ImagePath
```

Add before `ingest_image` function:

```python
@dataclass
class _PendingImage:
    image_path: Path
    image: Image.Image
    ocr_text: str
    caption: str
```

- [ ] **Step 2: Add `IngestionBatchAccumulator` class**

Add after `_PendingImage`:

```python
class IngestionBatchAccumulator:
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
        ExtractedImageText.create(ocr_text, caption)
        self.pending.append(_PendingImage(image_path, image, ocr_text, caption))

    def is_ready(self) -> bool:
        return len(self.pending) >= self.batch_size

    def flush(self) -> list[ImageDocument]:
        if not self.pending:
            return []
        pending = self.pending
        self.pending = []
        images = [p.image for p in pending]
        texts = [ExtractedImageText.create(p.ocr_text, p.caption).combined for p in pending]
        dense_vectors = self.encoder.encode_images(images)
        sparse_vectors = self.sparse.encode_texts(texts)
        documents: list[ImageDocument] = []
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

- [ ] **Step 3: Run existing tests to verify no regression**

Run: `uv run pytest tests/ -v`
Expected: All existing tests pass

- [ ] **Step 4: Commit**

```bash
git add scripts/ingest_images.py
git commit -m "feat: add IngestionBatchAccumulator class"
```

---

### Task 4: Rewrite ingest_images() to use IngestionBatchAccumulator

**Files:**
- Modify: `scripts/ingest_images.py`

- [ ] **Step 1: Replace the `ingest_images()` function body**

Replace the entire `ingest_images` function with:

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
    if batch_size < 1:
        raise ValueError("Batch size must be greater than zero")

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

    remaining = accumulator.flush()
    if remaining:
        image_index.upsert(remaining)
        documents.extend(remaining)

    return documents
```

- [ ] **Step 2: Update FakeDenseEncoder and FakeSparseEncoder in tests**

In `tests/test_scripts.py`, update the fake encoders to support batch methods:

```python
class FakeDenseEncoder:
    def encode_image(self, image_path: Path) -> DenseVector:
        return DenseVector.create([0.1, 0.2])

    def encode_images(self, images: list[Image.Image]) -> list[DenseVector]:
        return [DenseVector.create([0.1, 0.2]) for _ in images]

    def encode_text(self, text: str) -> DenseVector:
        return DenseVector.create([0.1, 0.2])


class FakeSparseEncoder:
    def encode_text(self, text: str) -> SparseVector:
        return SparseVector.create([1, 2], [0.3, 0.4])

    def encode_texts(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector.create([1, 2], [0.3, 0.4]) for _ in texts]
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_scripts.py -v`
Expected: `test_ingest_images_flushes_batches` passes with `[2, 1]` batch sizes

- [ ] **Step 4: Commit**

```bash
git add scripts/ingest_images.py tests/test_scripts.py
git commit -m "refactor: rewrite ingest_images to use IngestionBatchAccumulator"
```

---

### Task 5: Verify end-to-end and clean up

**Files:**
- All modified files

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 2: Verify CLI still works**

Run: `uv run python scripts/ingest_images.py --help`
Expected: Shows help with `--batch-size` option

- [ ] **Step 3: Final commit if needed**

```bash
git status
```
