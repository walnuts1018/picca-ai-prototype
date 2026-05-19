# Serverized Image Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current script-driven prototype with a service-oriented runtime composed of a gateway, model services, RabbitMQ, SeaweedFS, and Qdrant, while preserving weighted image search and adding upload/publish and search test clients.

**Architecture:** Introduce a `gateway` service that owns the external HTTP API, RabbitMQ consumption, SeaweedFS downloads, batching, and Qdrant access. Split model execution into `dense-service`, `sparse-service`, `ocr-service`, and `caption-service`, each exposing simple HTTP endpoints and choosing CPU or CUDA at deployment time.

**Tech Stack:** Python 3.12, FastAPI, Uvicorn, httpx, boto3, aio-pika or pika, Qdrant client, Pillow, existing transformer/Paddle model adapters, Docker Compose, Docker multi-stage builds

---

## File Structure

| Path | Action | Responsibility |
| --- | --- | --- |
| `pyproject.toml` | Modify | Add runtime dependencies for HTTP services, S3, RabbitMQ, and API clients |
| `src/picca_search/domain.py` | Modify | Add S3-key-based image identity and logical image source path handling |
| `src/picca_search/application.py` | Modify | Separate pure document-building and weighted search orchestration from CLI assumptions |
| `src/picca_search/infrastructure/qdrant_index.py` | Modify | Accept runtime search weights and keep diagnostics path |
| `src/picca_search/infrastructure/object_storage.py` | Create | Download SeaweedFS S3 objects to temp files |
| `src/picca_search/infrastructure/rabbitmq_queue.py` | Create | Consume and publish image job messages |
| `src/picca_search/infrastructure/model_client.py` | Create | HTTP clients for dense, sparse, OCR, and caption services |
| `src/picca_search/infrastructure/image_preprocessing.py` | Create | Move `prepare_inference_image()` out of CLI script for gateway reuse |
| `src/picca_search/gateway/schemas.py` | Create | FastAPI request/response schemas |
| `src/picca_search/gateway/search_api.py` | Create | Search route handlers and app factory |
| `src/picca_search/gateway/ingestion.py` | Create | Batch accumulation and ingestion orchestration |
| `src/picca_search/gateway/runtime.py` | Create | Startup wiring, background queue consumer, config loading |
| `src/picca_search/gateway/config.py` | Create | Environment-based settings for queue, S3, Qdrant, service URLs, batch limits |
| `src/picca_search/services/common.py` | Create | Shared FastAPI helpers for model services |
| `src/picca_search/services/dense_api.py` | Create | Dense image/text encoding HTTP service |
| `src/picca_search/services/sparse_api.py` | Create | Sparse text encoding HTTP service |
| `src/picca_search/services/ocr_api.py` | Create | OCR HTTP service |
| `src/picca_search/services/caption_api.py` | Create | Caption HTTP service |
| `scripts/run_gateway.py` | Create | Gateway composition root |
| `scripts/run_dense_service.py` | Create | Dense service composition root |
| `scripts/run_sparse_service.py` | Create | Sparse service composition root |
| `scripts/run_ocr_service.py` | Create | OCR service composition root |
| `scripts/run_caption_service.py` | Create | Caption service composition root |
| `scripts/publish_directory_to_queue.py` | Create | Bulk upload local images to SeaweedFS and publish RabbitMQ jobs |
| `scripts/search_api_client.py` | Create | Simple CLI client for `POST /search` |
| `compose.yaml` | Modify | Add RabbitMQ, SeaweedFS, gateway, and model services |
| `gateway.Dockerfile` | Create | Gateway image |
| `model.Dockerfile` | Create | Shared model-service image build with CPU/CUDA variants via build args |
| `README.md` | Modify | Update local run instructions and service topology |
| `tests/test_domain_runtime.py` | Create | S3 key identity and logical image path tests |
| `tests/test_qdrant_weights.py` | Create | Runtime weight override tests |
| `tests/test_gateway_api.py` | Create | Search API request/response behavior |
| `tests/test_gateway_ingestion.py` | Create | Batch accumulation and partial-failure behavior |
| `tests/test_model_clients.py` | Create | HTTP model client adapters |
| `tests/test_service_endpoints.py` | Create | Dense/sparse/OCR/caption endpoint behavior with fakes |
| `tests/test_publish_script.py` | Create | Upload-and-publish client behavior with fakes |

### Task 1: Refactor Domain Types for Service Runtime

**Files:**

- Modify: `src/picca_search/domain.py`
- Modify: `src/picca_search/application.py`
- Create: `tests/test_domain_runtime.py`

- [ ] **Step 1: Write the failing domain tests**

```python
from picca_search.domain import ImageId, ImageSourcePath


def test_image_id_from_object_key_preserves_key() -> None:
    image_id = ImageId.from_object_key("folder/example.jpg")
    assert image_id.value == "folder/example.jpg"


def test_image_source_path_accepts_s3_uri_without_local_file_check() -> None:
    source = ImageSourcePath.create("s3://images/folder/example.jpg")
    assert source.value == "s3://images/folder/example.jpg"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_domain_runtime.py -v`
Expected: FAIL with `AttributeError` or `ImportError` because `from_object_key` and `ImageSourcePath` do not exist.

- [ ] **Step 3: Implement minimal domain changes**

```python
@dataclass(frozen=True)
class ImageId:
    value: str

    @classmethod
    def from_object_key(cls, object_key: str) -> "ImageId":
        normalized = object_key.strip("/")
        if normalized == "":
            raise ValueError("Object key must not be blank")
        return cls(normalized)


@dataclass(frozen=True)
class ImageSourcePath:
    value: str

    @classmethod
    def create(cls, value: str) -> "ImageSourcePath":
        normalized = value.strip()
        if normalized == "":
            raise ValueError("Image source path must not be blank")
        return cls(normalized)
```

- [ ] **Step 4: Update `ImageDocument` payload generation to use logical source paths**

```python
@dataclass(frozen=True)
class ImageDocument:
    image_id: ImageId
    source_path: ImageSourcePath
    ...

    @property
    def payload(self) -> dict[str, str]:
        payload = {
            "path": self.source_path.value,
            "text": self.text,
        }
        ...
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_domain_runtime.py tests/test_weighted_multivector_search.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/picca_search/domain.py src/picca_search/application.py tests/test_domain_runtime.py tests/test_weighted_multivector_search.py
git commit -m "refactor: support object-key based image identity"
```

### Task 2: Add Runtime Weight Overrides to Qdrant Search

**Files:**

- Modify: `src/picca_search/infrastructure/qdrant_index.py`
- Create: `tests/test_qdrant_weights.py`

- [ ] **Step 1: Write the failing weight-override tests**

```python
def test_search_with_diagnostics_uses_runtime_weights() -> None:
    index = QdrantImageIndex(_FakeClient(), "images")
    diagnostics = index.search_with_diagnostics(
        query_dense=_dense([0.5, 0.6]),
        query_ocr_sparse=_sparse([3], [0.7]),
        query_florence_sparse=_sparse([4], [0.8]),
        limit=5,
        weights=(8.0, 1.0, 0.5),
    )
    assert diagnostics.fused[0].score == 8.0 / 2.0 + 1.0 / 2.0 + 0.5 / 3.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_qdrant_weights.py -v`
Expected: FAIL with `TypeError` because `weights` is not an accepted argument.

- [ ] **Step 3: Implement the minimal search override**

```python
DEFAULT_RRF_WEIGHTS = (DENSE_WEIGHT, OCR_WEIGHT, FLORENCE_WEIGHT)


def _resolve_weights(weights: tuple[float, float, float] | None) -> tuple[float, float, float]:
    return DEFAULT_RRF_WEIGHTS if weights is None else weights


def search(..., weights: tuple[float, float, float] | None = None) -> list[SearchResult]:
    resolved_weights = _resolve_weights(weights)
    ...
    query=models.RrfQuery(rrf=models.Rrf(k=RRF_K, weights=list(resolved_weights)))
```

- [ ] **Step 4: Thread the same override through diagnostics**

```python
def search_with_diagnostics(..., weights: tuple[float, float, float] | None = None) -> SearchDiagnostics:
    resolved_weights = _resolve_weights(weights)
    ...
    for results, weight in zip((dense, ocr, florence), resolved_weights, strict=True):
        ...
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_qdrant_weights.py tests/test_weighted_multivector_search.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/picca_search/infrastructure/qdrant_index.py tests/test_qdrant_weights.py tests/test_weighted_multivector_search.py
git commit -m "feat: allow runtime weight overrides in qdrant search"
```

### Task 3: Introduce Shared Image Preprocessing and Object Storage

**Files:**

- Create: `src/picca_search/infrastructure/image_preprocessing.py`
- Create: `src/picca_search/infrastructure/object_storage.py`
- Create: `tests/test_gateway_ingestion.py`

- [ ] **Step 1: Write the failing object storage and preprocessing tests**

```python
def test_download_image_writes_temp_file(tmp_path: Path) -> None:
    storage = SeaweedObjectStorage(fake_s3_client, bucket="images", temp_dir=tmp_path)
    local_path = storage.download_to_tempfile("folder/example.jpg")
    assert local_path.exists()


def test_prepare_inference_image_keeps_original_when_no_changes_needed(sample_image: Path) -> None:
    with prepare_inference_image(sample_image) as prepared:
        assert prepared == sample_image
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gateway_ingestion.py -v`
Expected: FAIL because `SeaweedObjectStorage` and shared `prepare_inference_image` module do not exist.

- [ ] **Step 3: Move preprocessing out of the CLI script**

```python
@contextmanager
def prepare_inference_image(image_path: Path) -> Iterator[Path]:
    with Image.open(image_path) as image:
        normalized_image = ImageOps.exif_transpose(image)
        ...
        yield image_path
```

- [ ] **Step 4: Implement the SeaweedFS S3 adapter**

```python
class SeaweedObjectStorage:
    def __init__(self, s3_client: Any, bucket: str, temp_dir: Path | None = None) -> None:
        self.s3_client = s3_client
        self.bucket = bucket
        self.temp_dir = temp_dir

    def download_to_tempfile(self, object_key: str) -> Path:
        target = tempfile.NamedTemporaryFile(delete=False, dir=self.temp_dir)
        self.s3_client.download_file(self.bucket, object_key, target.name)
        return Path(target.name)
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_gateway_ingestion.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/picca_search/infrastructure/image_preprocessing.py src/picca_search/infrastructure/object_storage.py tests/test_gateway_ingestion.py
git commit -m "feat: add shared image preprocessing and object storage adapter"
```

### Task 4: Build Dense and Sparse Model Services

**Files:**

- Create: `src/picca_search/services/common.py`
- Create: `src/picca_search/services/dense_api.py`
- Create: `src/picca_search/services/sparse_api.py`
- Create: `scripts/run_dense_service.py`
- Create: `scripts/run_sparse_service.py`
- Create: `tests/test_service_endpoints.py`

- [ ] **Step 1: Write the failing endpoint tests**

```python
def test_dense_text_batch_endpoint_returns_vectors(client) -> None:
    response = client.post("/encode/text-batch", json={"texts": ["a", "b"]})
    assert response.status_code == 200
    assert response.json()["vectors"][0] == [0.1, 0.2]


def test_sparse_text_batch_endpoint_returns_sparse_vectors(client) -> None:
    response = client.post("/encode/text-batch", json={"texts": ["hello"]})
    assert response.status_code == 200
    assert response.json()["vectors"][0]["indices"] == [1, 4]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_service_endpoints.py -k "dense or sparse" -v`
Expected: FAIL because the service apps do not exist.

- [ ] **Step 3: Implement shared FastAPI service helpers**

```python
class TextBatchRequest(BaseModel):
    texts: list[str]


def create_service_app(title: str) -> FastAPI:
    return FastAPI(title=title)
```

- [ ] **Step 4: Implement dense and sparse service apps**

```python
def create_dense_app(encoder: WaonSiglipEncoder) -> FastAPI:
    app = create_service_app("dense-service")

    @app.post("/encode/text-batch")
    def encode_text_batch(request: TextBatchRequest) -> dict[str, object]:
        return {"vectors": [list(encoder.encode_text(text).values) for text in request.texts]}

    return app
```

```python
def create_sparse_app(encoder: SpladeJapaneseSparseEncoder) -> FastAPI:
    app = create_service_app("sparse-service")

    @app.post("/encode/text-batch")
    def encode_text_batch(request: TextBatchRequest) -> dict[str, object]:
        return {
            "vectors": [
                {"indices": list(vector.indices), "values": list(vector.values)}
                for vector in encoder.encode_texts(request.texts)
            ]
        }

    return app
```

- [ ] **Step 5: Add composition roots**

```python
if __name__ == "__main__":
    encoder = WaonSiglipEncoder(device=os.getenv("MODEL_DEVICE"))
    uvicorn.run(create_dense_app(encoder), host="0.0.0.0", port=8001)
```

- [ ] **Step 6: Run tests to verify pass**

Run: `uv run pytest tests/test_service_endpoints.py -k "dense or sparse" -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/picca_search/services/common.py src/picca_search/services/dense_api.py src/picca_search/services/sparse_api.py scripts/run_dense_service.py scripts/run_sparse_service.py tests/test_service_endpoints.py
git commit -m "feat: add dense and sparse model services"
```

### Task 5: Build OCR and Caption Model Services

**Files:**

- Create: `src/picca_search/services/ocr_api.py`
- Create: `src/picca_search/services/caption_api.py`
- Create: `scripts/run_ocr_service.py`
- Create: `scripts/run_caption_service.py`
- Modify: `tests/test_service_endpoints.py`

- [ ] **Step 1: Write the failing OCR/caption endpoint tests**

```python
def test_ocr_endpoint_returns_text(client, sample_upload) -> None:
    response = client.post("/extract", files={"image": sample_upload})
    assert response.status_code == 200
    assert response.json() == {"text": "receipt total"}


def test_caption_endpoint_returns_text(client, sample_upload) -> None:
    response = client.post("/caption", files={"image": sample_upload})
    assert response.status_code == 200
    assert response.json() == {"text": "赤い鳥居"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_service_endpoints.py -k "ocr or caption" -v`
Expected: FAIL because OCR and caption apps do not exist.

- [ ] **Step 3: Implement file-upload based OCR/caption apps**

```python
@app.post("/extract")
async def extract(file: UploadFile) -> dict[str, str]:
    temp_path = await write_upload_to_tempfile(file)
    try:
        return {"text": extractor.extract_text(temp_path)}
    finally:
        temp_path.unlink(missing_ok=True)
```

```python
@app.post("/caption")
async def caption(file: UploadFile) -> dict[str, str]:
    temp_path = await write_upload_to_tempfile(file)
    try:
        return {"text": captioner.caption(temp_path)}
    finally:
        temp_path.unlink(missing_ok=True)
```

- [ ] **Step 4: Add composition roots with device env selection**

```python
if __name__ == "__main__":
    extractor = PaddleOcrVlTextExtractor()
    uvicorn.run(create_ocr_app(extractor), host="0.0.0.0", port=8003)
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_service_endpoints.py -k "ocr or caption" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/picca_search/services/ocr_api.py src/picca_search/services/caption_api.py scripts/run_ocr_service.py scripts/run_caption_service.py tests/test_service_endpoints.py
git commit -m "feat: add ocr and caption model services"
```

### Task 6: Add Gateway HTTP Clients and Search API

**Files:**

- Create: `src/picca_search/infrastructure/model_client.py`
- Create: `src/picca_search/gateway/schemas.py`
- Create: `src/picca_search/gateway/search_api.py`
- Create: `tests/test_model_clients.py`
- Create: `tests/test_gateway_api.py`

- [ ] **Step 1: Write the failing client and API tests**

```python
def test_dense_client_encodes_text_batch(httpx_mock) -> None:
    httpx_mock.add_response(json={"vectors": [[0.1, 0.2]]})
    client = DenseModelClient("http://dense:8001")
    vectors = client.encode_texts(["hello"])
    assert vectors[0].values == (0.1, 0.2)


def test_search_route_passes_runtime_weights(fake_dependencies) -> None:
    response = client.post("/search", json={"query": "torii", "dense_weight": 9.0})
    assert response.status_code == 200
    assert response.json()["weights"]["dense"] == 9.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_model_clients.py tests/test_gateway_api.py -v`
Expected: FAIL because clients and search app do not exist.

- [ ] **Step 3: Implement model HTTP clients**

```python
class DenseModelClient:
    def __init__(self, base_url: str, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=60.0)

    def encode_texts(self, texts: list[str]) -> list[DenseVector]:
        response = self.client.post(f"{self.base_url}/encode/text-batch", json={"texts": texts})
        response.raise_for_status()
        return [DenseVector.create(values) for values in response.json()["vectors"]]
```

- [ ] **Step 4: Implement the search FastAPI app**

```python
@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, dependencies: GatewayDependencies = Depends(...)) -> SearchResponse:
    dense_vector = dependencies.dense_client.encode_texts([request.query])[0]
    sparse_vector = dependencies.sparse_client.encode_texts([request.query])[0]
    diagnostics = dependencies.index.search_with_diagnostics(
        query_dense=dense_vector,
        query_ocr_sparse=sparse_vector,
        query_florence_sparse=sparse_vector,
        limit=request.limit,
        weights=request.weights_tuple(),
    )
    return SearchResponse.from_diagnostics(request, diagnostics)
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_model_clients.py tests/test_gateway_api.py tests/test_qdrant_weights.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/picca_search/infrastructure/model_client.py src/picca_search/gateway/schemas.py src/picca_search/gateway/search_api.py tests/test_model_clients.py tests/test_gateway_api.py
git commit -m "feat: add gateway search api and model clients"
```

### Task 7: Implement Queue-Driven Ingestion in the Gateway

**Files:**

- Create: `src/picca_search/infrastructure/rabbitmq_queue.py`
- Create: `src/picca_search/gateway/ingestion.py`
- Create: `src/picca_search/gateway/config.py`
- Create: `src/picca_search/gateway/runtime.py`
- Create: `scripts/run_gateway.py`
- Modify: `tests/test_gateway_ingestion.py`

- [ ] **Step 1: Write the failing ingestion runtime tests**

```python
def test_ingestion_batch_flushes_on_size(fake_dependencies) -> None:
    ingestor = GatewayIngestionService(..., max_batch_size=2, max_batch_wait_ms=1000)
    ingestor.enqueue("folder/a.jpg")
    ingestor.enqueue("folder/b.jpg")
    assert fake_index.upserted_ids == ["folder/a.jpg", "folder/b.jpg"]


def test_non_retryable_failure_is_not_requeued(fake_dependencies) -> None:
    result = service.process_message({"image_id": "missing.jpg"})
    assert result.action == "dead-letter"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gateway_ingestion.py -v`
Expected: FAIL because queue runtime and ingestion service do not exist.

- [ ] **Step 3: Implement RabbitMQ message contract and gateway ingestion service**

```python
@dataclass(frozen=True)
class ImageJobMessage:
    image_id: str

    @classmethod
    def from_body(cls, body: bytes) -> "ImageJobMessage":
        payload = json.loads(body)
        return cls(image_id=payload["image_id"])
```

```python
class GatewayIngestionService:
    def process_batch(self, image_ids: list[str]) -> BatchResult:
        downloaded = [self.storage.download_to_tempfile(image_id) for image_id in image_ids]
        ...
        self.index.upsert(documents)
        return BatchResult(acked=image_ids, dead_lettered=failures)
```

- [ ] **Step 4: Implement runtime wiring and background consumer**

```python
def create_gateway_runtime(settings: GatewaySettings) -> FastAPI:
    app = create_search_app(...)

    @app.on_event("startup")
    async def start_consumer() -> None:
        app.state.consumer_task = asyncio.create_task(run_queue_consumer(...))

    return app
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_gateway_ingestion.py tests/test_gateway_api.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/picca_search/infrastructure/rabbitmq_queue.py src/picca_search/gateway/ingestion.py src/picca_search/gateway/config.py src/picca_search/gateway/runtime.py scripts/run_gateway.py tests/test_gateway_ingestion.py
git commit -m "feat: add queue-driven gateway ingestion runtime"
```

### Task 8: Add Compose, Docker, and Test Utility Scripts

**Files:**

- Modify: `compose.yaml`
- Create: `gateway.Dockerfile`
- Create: `model.Dockerfile`
- Create: `scripts/publish_directory_to_queue.py`
- Create: `scripts/search_api_client.py`
- Create: `tests/test_publish_script.py`
- Modify: `README.md`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write the failing utility-script tests**

```python
def test_publish_script_uploads_and_publishes(fake_s3, fake_queue, tmp_path: Path) -> None:
    image = tmp_path / "a.jpg"
    image.write_bytes(b"jpg")
    main(["--image-dir", str(tmp_path)])
    assert fake_s3.uploaded_keys == ["a.jpg"]
    assert fake_queue.messages == [{"image_id": "a.jpg"}]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_publish_script.py -v`
Expected: FAIL because the script and new dependencies are missing.

- [ ] **Step 3: Add runtime dependencies**

```toml
dependencies = [
  ...,
  "fastapi>=0.115.0",
  "uvicorn>=0.30.0",
  "httpx>=0.27.0",
  "boto3>=1.35.0",
  "pika>=1.3.2",
  "pydantic>=2.8.0",
  "python-multipart>=0.0.9",
]
```

- [ ] **Step 4: Implement Compose and Dockerfiles**

```yaml
services:
  rabbitmq:
    image: rabbitmq:3.13-management
  seaweedfs-master:
    image: chrislusf/seaweedfs:3.89
  gateway:
    build:
      context: .
      dockerfile: gateway.Dockerfile
```

```dockerfile
FROM python:3.12-slim AS builder
RUN --mount=type=cache,target=/root/.cache/uv pip install uv
...
FROM python:3.12-slim AS runtime
COPY --from=builder /app /app
CMD ["uv", "run", "python", "scripts/run_gateway.py"]
```

- [ ] **Step 5: Implement publish and search client scripts**

```python
def publish_image_jobs(image_dir: Path, bucket: str, prefix: str = "") -> None:
    for path in sorted(image_dir.iterdir()):
        object_key = f"{prefix}{path.name}"
        s3.upload_file(str(path), bucket, object_key)
        queue.publish({"image_id": object_key})
```

```python
response = httpx.post(
    f"{base_url}/search",
    json={"query": args.query, "limit": args.limit, "dense_weight": args.dense_weight},
)
```

- [ ] **Step 6: Run tests to verify pass**

Run: `uv run pytest tests/test_publish_script.py tests/test_gateway_api.py tests/test_service_endpoints.py -v`
Expected: PASS

- [ ] **Step 7: Perform a broader verification run**

Run: `uv run pytest -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml compose.yaml gateway.Dockerfile model.Dockerfile scripts/publish_directory_to_queue.py scripts/search_api_client.py README.md tests/test_publish_script.py
git commit -m "feat: add deployment assets and test utility clients"
```

## Self-Review

- Spec coverage: domain identity/path changes, runtime weight overrides, queue ingestion, SeaweedFS, RabbitMQ, HTTP search API, model services, Docker/Compose, and test utility scripts are all covered by Tasks 1-8.
- Placeholder scan: no `TODO`, `TBD`, or “implement later” placeholders remain.
- Type consistency: the plan consistently uses `ImageId.from_object_key`, `ImageSourcePath`, `weights=(dense, ocr, florence)`, and the `gateway`/model-service naming from the spec.
