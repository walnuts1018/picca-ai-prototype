# Debug Web Application Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a simple Go-based debug web application for image upload, queue publish, search, image proxy, and image-level ingestion status tracking, while extending the Python gateway to publish result events and expose score breakdowns.

**Architecture:** Add a new `debug-web` Go service that owns image status in sqlite, uploads to S3, publishes jobs to RabbitMQ, consumes result events from RabbitMQ, proxies images from S3, and calls the gateway search API. Extend the existing Python gateway to publish image-level ingestion result events and to include per-result score breakdown data in `POST /search` responses when diagnostics are requested.

**Tech Stack:** Go 1.26, `net/http`, `html/template`, `database/sql` with `modernc.org/sqlite`, `aws-sdk-go-v2`, `amqp091-go`, Python 3.12, FastAPI, boto3, pika, pytest, Docker Compose

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `debug-web/go.mod` | Create | Standalone Go module for the debug web app |
| `debug-web/go.sum` | Create | Go dependency lockfile |
| `debug-web/cmd/debug-web/main.go` | Create | Process entrypoint and lifecycle wiring |
| `debug-web/internal/config/config.go` | Create | Environment-driven settings |
| `debug-web/internal/storage/sqlite.go` | Create | Sqlite schema and image status repository |
| `debug-web/internal/storage/sqlite_test.go` | Create | Repository unit tests |
| `debug-web/internal/queue/messages.go` | Create | RabbitMQ job/result message contracts |
| `debug-web/internal/queue/rabbitmq.go` | Create | Publisher and result consumer |
| `debug-web/internal/queue/rabbitmq_test.go` | Create | Queue message parsing tests |
| `debug-web/internal/s3client/client.go` | Create | S3 upload/download/proxy client |
| `debug-web/internal/search/client.go` | Create | Gateway search client and response types |
| `debug-web/internal/search/client_test.go` | Create | Search client unit tests |
| `debug-web/internal/web/templates/index.html` | Create | Single-page debug UI template |
| `debug-web/internal/web/handlers.go` | Create | HTTP handlers for upload/search/status/image proxy |
| `debug-web/internal/web/handlers_test.go` | Create | HTTP handler tests |
| `debug-web/internal/app/app.go` | Create | App assembly and background consumer startup |
| `debug-web.Dockerfile` | Create | Container image for the Go debug web app |
| `compose.yaml` | Modify | Add `debug-web` service and result queue env |
| `src/picca_search/infrastructure/rabbitmq_queue.py` | Modify | Add result event message/publisher support |
| `src/picca_search/gateway/config.py` | Modify | Add result queue setting |
| `src/picca_search/gateway/ingestion.py` | Modify | Return image-level status details from batch processing |
| `src/picca_search/gateway/runtime.py` | Modify | Publish `processing/indexed/failed` events |
| `src/picca_search/gateway/schemas.py` | Modify | Add score breakdown response types |
| `src/picca_search/gateway/search_api.py` | Modify | Populate score breakdowns in search responses |
| `tests/test_gateway_result_events.py` | Create | Gateway event publication and payload tests |
| `tests/test_gateway_search_breakdown.py` | Create | Search breakdown calculation tests |
| `README.md` | Modify | Document debug web usage and configuration |

### Task 1: Add Gateway Search Breakdown Response

**Files:**
- Modify: `src/picca_search/gateway/schemas.py`
- Modify: `src/picca_search/gateway/search_api.py`
- Test: `tests/test_gateway_search_breakdown.py`

- [ ] **Step 1: Write the failing breakdown response tests**

```python
from picca_search.gateway.search_api import create_search_app, GatewaySearchDependencies
from picca_search.domain import DenseVector, SparseVector, ImageId, SearchResult
from fastapi.testclient import TestClient


class _FakeDense:
    def encode_texts(self, texts: list[str]) -> list[DenseVector]:
        return [DenseVector.create([1.0, 0.0])]


class _FakeSparse:
    def encode_texts(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector.create([1], [1.0])]


class _FakeIndex:
    def search_with_diagnostics(self, **kwargs):
        class _Ranked:
            def __init__(self, image_id: str, rank: int, score: float, payload: dict[str, object]) -> None:
                self.image_id = ImageId(image_id)
                self.rank = rank
                self.score = score
                self.payload = payload

        payload = {"path": "s3://images/a.jpg", "text": "torii", "caption": "red gate"}
        fused = [SearchResult(image_id=ImageId("a.jpg"), score=3.5, payload=payload)]
        dense = [_Ranked("a.jpg", 1, 0.91, payload)]
        ocr = [_Ranked("a.jpg", 2, 0.52, payload)]
        florence = []
        return type("Diagnostics", (), {"fused": fused, "dense": dense, "ocr": ocr, "florence": florence})()


def test_search_with_diagnostics_returns_score_breakdown() -> None:
    app = create_search_app(
        GatewaySearchDependencies(dense_client=_FakeDense(), sparse_client=_FakeSparse(), index=_FakeIndex())
    )
    client = TestClient(app)
    response = client.post("/search", json={"query": "torii", "include_diagnostics": True})
    body = response.json()
    result = body["results"][0]
    assert result["score"] == 3.5
    assert result["score_breakdown"]["dense_rank"] == 1
    assert result["score_breakdown"]["ocr_rank"] == 2
    assert result["score_breakdown"]["florence_rank"] is None
    assert result["score_breakdown"]["dense_score"] == 2.0
    assert result["score_breakdown"]["ocr_score"] == 2.0 / 3.0
    assert result["score_breakdown"]["florence_score"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gateway_search_breakdown.py -v`  
Expected: FAIL because `score_breakdown` is not present in the search response model.

- [ ] **Step 3: Write minimal implementation**

```python
# src/picca_search/gateway/schemas.py
class SearchScoreBreakdown(BaseModel):
    dense_score: float
    ocr_score: float
    florence_score: float
    dense_rank: int | None = None
    ocr_rank: int | None = None
    florence_rank: int | None = None


class SearchResultPayload(BaseModel):
    image_id: str
    score: float
    path: str
    text: str
    ocr_text: str | None = None
    caption: str | None = None
    score_breakdown: SearchScoreBreakdown | None = None
```

```python
# src/picca_search/gateway/search_api.py
def _result_payload(result: Any, breakdown: SearchScoreBreakdown | None = None) -> SearchResultPayload:
    payload = dict(result.payload)
    return SearchResultPayload(
        image_id=result.image_id.value,
        score=result.score,
        path=str(payload.get("path", "")),
        text=str(payload.get("text", "")),
        ocr_text=(str(payload["ocr_text"]) if "ocr_text" in payload else None),
        caption=(str(payload["caption"]) if "caption" in payload else None),
        score_breakdown=breakdown,
    )


def _build_breakdown_maps(
    diagnostics: Any, weights: tuple[float, float, float]
) -> dict[str, SearchScoreBreakdown]:
    breakdowns: dict[str, dict[str, float | int | None]] = {}
    for key, results, weight in (
        ("dense", diagnostics.dense, weights[0]),
        ("ocr", diagnostics.ocr, weights[1]),
        ("florence", diagnostics.florence, weights[2]),
    ):
        for result in results:
            current = breakdowns.setdefault(
                result.image_id.value,
                {
                    "dense_score": 0.0,
                    "ocr_score": 0.0,
                    "florence_score": 0.0,
                    "dense_rank": None,
                    "ocr_rank": None,
                    "florence_rank": None,
                },
            )
            current[f"{key}_rank"] = result.rank
            current[f"{key}_score"] = weight / (1 + result.rank)
    return {
        image_id: SearchScoreBreakdown(**values)
        for image_id, values in breakdowns.items()
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_gateway_search_breakdown.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/picca_search/gateway/schemas.py src/picca_search/gateway/search_api.py tests/test_gateway_search_breakdown.py
git commit -m "feat: add score breakdowns to gateway search responses"
```

### Task 2: Add Gateway Result Event Messages

**Files:**
- Modify: `src/picca_search/infrastructure/rabbitmq_queue.py`
- Modify: `src/picca_search/gateway/config.py`
- Test: `tests/test_gateway_result_events.py`

- [ ] **Step 1: Write the failing result event message tests**

```python
from picca_search.infrastructure.rabbitmq_queue import ImageJobResultMessage


def test_result_message_serializes_failed_event() -> None:
    message = ImageJobResultMessage(
        image_id="debug/a.jpg",
        status="failed",
        occurred_at="2026-05-20T12:00:00Z",
        error_message="caption timeout",
    )
    parsed = ImageJobResultMessage.from_body(message.to_body())
    assert parsed.image_id == "debug/a.jpg"
    assert parsed.status == "failed"
    assert parsed.occurred_at == "2026-05-20T12:00:00Z"
    assert parsed.error_message == "caption timeout"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gateway_result_events.py -v`  
Expected: FAIL because `ImageJobResultMessage` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# src/picca_search/infrastructure/rabbitmq_queue.py
@dataclass(frozen=True)
class ImageJobResultMessage:
    image_id: str
    status: str
    occurred_at: str
    error_message: str | None = None

    @classmethod
    def from_body(cls, body: bytes) -> "ImageJobResultMessage":
        payload = json.loads(body.decode("utf-8"))
        return cls(
            image_id=str(payload["image_id"]).strip(),
            status=str(payload["status"]).strip(),
            occurred_at=str(payload["occurred_at"]).strip(),
            error_message=(None if payload.get("error_message") in (None, "") else str(payload["error_message"])),
        )

    def to_body(self) -> bytes:
        return json.dumps(
            {
                "image_id": self.image_id,
                "status": self.status,
                "occurred_at": self.occurred_at,
                "error_message": self.error_message,
            },
            ensure_ascii=False,
        ).encode("utf-8")
```

```python
# src/picca_search/gateway/config.py
@dataclass(frozen=True)
class GatewaySettings:
    ...
    rabbitmq_result_queue: str = "image_job_results"

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        return cls(
            ...
            rabbitmq_result_queue=os.getenv("RABBITMQ_RESULT_QUEUE", "image_job_results"),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_gateway_result_events.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/picca_search/infrastructure/rabbitmq_queue.py src/picca_search/gateway/config.py tests/test_gateway_result_events.py
git commit -m "feat: add gateway result event message types"
```

### Task 3: Publish Image-Level Gateway Result Events

**Files:**
- Modify: `src/picca_search/gateway/ingestion.py`
- Modify: `src/picca_search/gateway/runtime.py`
- Modify: `src/picca_search/infrastructure/rabbitmq_queue.py`
- Test: `tests/test_gateway_result_events.py`

- [ ] **Step 1: Write the failing gateway publish tests**

```python
from picca_search.gateway.ingestion import IngestionOutcome


def test_ingestion_outcome_keeps_image_level_statuses() -> None:
    outcome = IngestionOutcome(
        acked_delivery_tags=[1],
        requeue_delivery_tags=[],
        dead_letter_delivery_tags=[2],
        events=[
            {"image_id": "debug/a.jpg", "status": "indexed", "error_message": None},
            {"image_id": "debug/b.jpg", "status": "failed", "error_message": "unsupported image"},
        ],
    )
    assert outcome.events[0]["status"] == "indexed"
    assert outcome.events[1]["error_message"] == "unsupported image"
```

```python
def test_runtime_publishes_result_events(monkeypatch) -> None:
    published: list[tuple[str, str]] = []

    class _FakePublisher:
        def publish_result(self, queue_name, message):
            published.append((queue_name, message.status))

    # runtime helper will be factored so the test can call it directly
    _publish_ingestion_events(
        publisher=_FakePublisher(),
        queue_name="image_job_results",
        events=[
            {"image_id": "debug/a.jpg", "status": "processing", "error_message": None},
            {"image_id": "debug/a.jpg", "status": "indexed", "error_message": None},
        ],
    )
    assert published == [("image_job_results", "processing"), ("image_job_results", "indexed")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gateway_result_events.py -v`  
Expected: FAIL because `IngestionOutcome.events` and `_publish_ingestion_events` do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# src/picca_search/gateway/ingestion.py
@dataclass(frozen=True)
class IngestionEvent:
    image_id: str
    status: str
    error_message: str | None = None


@dataclass(frozen=True)
class IngestionOutcome:
    acked_delivery_tags: list[int]
    requeue_delivery_tags: list[int]
    dead_letter_delivery_tags: list[int]
    events: list[IngestionEvent]
```

```python
# src/picca_search/gateway/runtime.py
def _publish_ingestion_events(publisher: RabbitMqImageJobQueue, queue_name: str, events: list[IngestionEvent]) -> None:
    occurred_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    for event in events:
        publisher.publish_result(
            queue_name,
            ImageJobResultMessage(
                image_id=event.image_id,
                status=event.status,
                occurred_at=occurred_at,
                error_message=event.error_message,
            ),
        )
```

```python
# src/picca_search/infrastructure/rabbitmq_queue.py
def publish_result(self, queue_name: str, message: ImageJobResultMessage) -> None:
    self.channel.queue_declare(queue=queue_name, durable=True)
    self.channel.basic_publish(
        exchange="",
        routing_key=queue_name,
        body=message.to_body(),
        properties=pika.BasicProperties(delivery_mode=2),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_gateway_result_events.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/picca_search/gateway/ingestion.py src/picca_search/gateway/runtime.py src/picca_search/infrastructure/rabbitmq_queue.py tests/test_gateway_result_events.py
git commit -m "feat: publish image-level ingestion result events"
```

### Task 4: Scaffold Debug Web Module and Repository

**Files:**
- Create: `debug-web/go.mod`
- Create: `debug-web/internal/config/config.go`
- Create: `debug-web/internal/storage/sqlite.go`
- Create: `debug-web/internal/storage/sqlite_test.go`
- Create: `debug-web/internal/queue/messages.go`

- [ ] **Step 1: Write the failing Go repository tests**

```go
package storage

import "testing"

func TestRepositoryUpsertAndList(t *testing.T) {
	repo, err := Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	err = repo.UpsertQueued(ImageRecord{
		ImageID:   "debug/a.jpg",
		ObjectKey: "debug/a.jpg",
		Filename:  "a.jpg",
	})
	if err != nil {
		t.Fatal(err)
	}
	err = repo.ApplyResult("debug/a.jpg", "indexed", "", "2026-05-20T12:00:00Z")
	if err != nil {
		t.Fatal(err)
	}
	rows, err := repo.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0].Status != "indexed" {
		t.Fatalf("unexpected rows: %#v", rows)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd debug-web && go test ./internal/storage -v`  
Expected: FAIL because the module and repository do not exist.

- [ ] **Step 3: Write minimal implementation**

```go
module github.com/walnuts1018/picca-ai-prototype/debug-web

go 1.26

require modernc.org/sqlite v1.39.0
```

```go
package storage

type ImageRecord struct {
	ImageID     string
	ObjectKey   string
	Filename    string
	Status      string
	ErrorMessage string
	UploadedAt  string
	LastEventAt string
	AttemptCount int
}
```

```go
func (r *Repository) UpsertQueued(record ImageRecord) error {
	_, err := r.db.Exec(`
		INSERT INTO image_records (image_id, object_key, filename, status, uploaded_at, last_event_at, attempt_count)
		VALUES (?, ?, ?, 'queued', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0)
		ON CONFLICT(image_id) DO UPDATE SET status='queued', last_event_at=CURRENT_TIMESTAMP
	`, record.ImageID, record.ObjectKey, record.Filename)
	return err
}
```

```go
func (r *Repository) ApplyResult(imageID string, status string, errorMessage string, occurredAt string) error {
	_, err := r.db.Exec(`
		UPDATE image_records
		SET status = ?, error_message = ?, last_event_at = ?, attempt_count = attempt_count + CASE WHEN status = 'processing' THEN 1 ELSE 0 END
		WHERE image_id = ?
	`, status, errorMessage, occurredAt, imageID)
	return err
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd debug-web && go test ./internal/storage -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add debug-web/go.mod debug-web/internal/config/config.go debug-web/internal/storage/sqlite.go debug-web/internal/storage/sqlite_test.go debug-web/internal/queue/messages.go
git commit -m "feat: scaffold debug web module and sqlite repository"
```

### Task 5: Add Debug Web Search Client and Queue Consumer

**Files:**
- Create: `debug-web/internal/search/client.go`
- Create: `debug-web/internal/search/client_test.go`
- Create: `debug-web/internal/queue/rabbitmq.go`
- Create: `debug-web/internal/queue/rabbitmq_test.go`

- [ ] **Step 1: Write the failing Go client tests**

```go
package search

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestClientSearchParsesBreakdown(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"query":"torii","limit":10,"weights":{"dense":4,"ocr":2,"florence":1},"results":[{"image_id":"debug/a.jpg","score":3.5,"path":"s3://images/debug/a.jpg","text":"torii","caption":"red gate","score_breakdown":{"dense_score":2,"ocr_score":0.6666667,"florence_score":0,"dense_rank":1,"ocr_rank":2,"florence_rank":null}}],"diagnostics":{}}`))
	}))
	defer server.Close()
	client := New(server.URL)
	resp, err := client.Search("torii", 10)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Results[0].ScoreBreakdown.DenseRank != 1 {
		t.Fatalf("unexpected breakdown: %#v", resp.Results[0].ScoreBreakdown)
	}
}
```

```go
package queue

import "testing"

func TestResultMessageDecode(t *testing.T) {
	body := []byte(`{"image_id":"debug/a.jpg","status":"failed","occurred_at":"2026-05-20T12:00:00Z","error_message":"ocr failed"}`)
	msg, err := ParseResultMessage(body)
	if err != nil {
		t.Fatal(err)
	}
	if msg.Status != "failed" || msg.ErrorMessage != "ocr failed" {
		t.Fatalf("unexpected msg: %#v", msg)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd debug-web && go test ./internal/search ./internal/queue -v`  
Expected: FAIL because the search client and queue parser do not exist.

- [ ] **Step 3: Write minimal implementation**

```go
package search

type ScoreBreakdown struct {
	DenseScore    float64 `json:"dense_score"`
	OCRScore      float64 `json:"ocr_score"`
	FlorenceScore float64 `json:"florence_score"`
	DenseRank     *int    `json:"dense_rank"`
	OCRRank       *int    `json:"ocr_rank"`
	FlorenceRank  *int    `json:"florence_rank"`
}
```

```go
func (c *Client) Search(query string, limit int) (*Response, error) {
	body := map[string]any{"query": query, "limit": limit, "include_diagnostics": true}
	...
}
```

```go
package queue

type ResultMessage struct {
	ImageID      string `json:"image_id"`
	Status       string `json:"status"`
	OccurredAt   string `json:"occurred_at"`
	ErrorMessage string `json:"error_message"`
}

func ParseResultMessage(body []byte) (ResultMessage, error) {
	var msg ResultMessage
	err := json.Unmarshal(body, &msg)
	return msg, err
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd debug-web && go test ./internal/search ./internal/queue -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add debug-web/internal/search/client.go debug-web/internal/search/client_test.go debug-web/internal/queue/rabbitmq.go debug-web/internal/queue/rabbitmq_test.go
git commit -m "feat: add debug web search client and queue parsing"
```

### Task 6: Add Debug Web HTTP Handlers and Template

**Files:**
- Create: `debug-web/internal/s3client/client.go`
- Create: `debug-web/internal/web/templates/index.html`
- Create: `debug-web/internal/web/handlers.go`
- Create: `debug-web/internal/web/handlers_test.go`
- Create: `debug-web/internal/app/app.go`
- Create: `debug-web/cmd/debug-web/main.go`

- [ ] **Step 1: Write the failing HTTP handler tests**

```go
package web

import (
	"bytes"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestUploadHandlerStoresQueuedRecords(t *testing.T) {
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, _ := writer.CreateFormFile("images", "a.jpg")
	_, _ = part.Write([]byte("fake-image"))
	_ = writer.Close()

	server := NewServer(FakeDependencies())
	req := httptest.NewRequest(http.MethodPost, "/uploads", &body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusSeeOther {
		t.Fatalf("unexpected status: %d", rec.Code)
	}
}
```

```go
func TestImageProxyReturnsBytes(t *testing.T) {
	server := NewServer(FakeDependenciesWithImage("debug/a.jpg", []byte("img")))
	req := httptest.NewRequest(http.MethodGet, "/images/debug/a.jpg", nil)
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("unexpected status: %d", rec.Code)
	}
	if rec.Body.String() != "img" {
		t.Fatalf("unexpected body: %q", rec.Body.String())
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd debug-web && go test ./internal/web -v`  
Expected: FAIL because the HTTP server and handlers do not exist.

- [ ] **Step 3: Write minimal implementation**

```go
// handlers.go
func NewServer(deps Dependencies) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /", deps.handleIndex)
	mux.HandleFunc("POST /uploads", deps.handleUploads)
	mux.HandleFunc("GET /uploads/statuses", deps.handleStatuses)
	mux.HandleFunc("POST /search", deps.handleSearch)
	mux.HandleFunc("GET /images/", deps.handleImageProxy)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	return mux
}
```

```go
func (d Dependencies) handleUploads(w http.ResponseWriter, r *http.Request) {
	files := r.MultipartForm.File["images"]
	for _, fileHeader := range files {
		objectKey := d.KeyFactory(fileHeader.Filename)
		_ = d.Repository.UpsertQueued(storage.ImageRecord{ImageID: objectKey, ObjectKey: objectKey, Filename: fileHeader.Filename})
		_ = d.JobPublisher.PublishJob(r.Context(), queue.JobMessage{ImageID: objectKey})
	}
	http.Redirect(w, r, "/", http.StatusSeeOther)
}
```

```go
func (d Dependencies) handleImageProxy(w http.ResponseWriter, r *http.Request) {
	imageID := strings.TrimPrefix(r.URL.Path, "/images/")
	contentType, body, err := d.ImageStore.Get(r.Context(), imageID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", contentType)
	_, _ = w.Write(body)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd debug-web && go test ./internal/web -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add debug-web/internal/s3client/client.go debug-web/internal/web/templates/index.html debug-web/internal/web/handlers.go debug-web/internal/web/handlers_test.go debug-web/internal/app/app.go debug-web/cmd/debug-web/main.go
git commit -m "feat: add debug web http handlers and ui"
```

### Task 7: Wire Result Consumer, Container, and Compose

**Files:**
- Create: `debug-web.Dockerfile`
- Modify: `compose.yaml`
- Modify: `README.md`
- Modify: `debug-web/internal/app/app.go`

- [ ] **Step 1: Write the failing app wiring tests**

```go
package app

import "testing"

func TestAppCanApplyResultMessage(t *testing.T) {
	repo := NewFakeRepository()
	err := applyResultMessage(repo, queue.ResultMessage{
		ImageID: "debug/a.jpg",
		Status: "indexed",
		OccurredAt: "2026-05-20T12:00:00Z",
	})
	if err != nil {
		t.Fatal(err)
	}
	if repo.LastStatus != "indexed" {
		t.Fatalf("unexpected status: %s", repo.LastStatus)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd debug-web && go test ./internal/app -v`  
Expected: FAIL because result consumer wiring does not exist.

- [ ] **Step 3: Write minimal implementation**

```go
// internal/app/app.go
func applyResultMessage(repo Repository, msg queue.ResultMessage) error {
	return repo.ApplyResult(msg.ImageID, msg.Status, msg.ErrorMessage, msg.OccurredAt)
}
```

```dockerfile
FROM golang:1.26 AS builder
WORKDIR /src
COPY debug-web/go.mod debug-web/go.sum ./debug-web/
WORKDIR /src/debug-web
RUN go mod download
COPY debug-web /src/debug-web
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o /out/debug-web ./cmd/debug-web

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=builder /out/debug-web /app/debug-web
EXPOSE 8080
CMD ["/app/debug-web"]
```

```yaml
# compose.yaml
  debug-web:
    build:
      context: .
      dockerfile: debug-web.Dockerfile
    depends_on:
      - rabbitmq
      - seaweedfs
      - gateway
    environment:
      DEBUG_WEB_HOST: 0.0.0.0
      DEBUG_WEB_PORT: 8080
      RABBITMQ_URL: amqp://guest:guest@rabbitmq:5672/%2F
      RABBITMQ_JOB_QUEUE: image_jobs
      RABBITMQ_RESULT_QUEUE: image_job_results
      S3_ENDPOINT_URL: http://seaweedfs:8333
      S3_BUCKET: images
      S3_ACCESS_KEY_ID: seaweedfs
      S3_SECRET_ACCESS_KEY: seaweedfs
      GATEWAY_BASE_URL: http://gateway:8000
      SQLITE_PATH: /data/debug-web.sqlite
    volumes:
      - ./data/debug-web:/data
    ports:
      - "8080:8080"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd debug-web && go test ./internal/app -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add debug-web/internal/app/app.go debug-web.Dockerfile compose.yaml README.md
git commit -m "feat: wire debug web runtime and container setup"
```

### Task 8: Run Focused Verification

**Files:**
- Test: `tests/test_gateway_result_events.py`
- Test: `tests/test_gateway_search_breakdown.py`
- Test: `debug-web/internal/storage/sqlite_test.go`
- Test: `debug-web/internal/search/client_test.go`
- Test: `debug-web/internal/queue/rabbitmq_test.go`
- Test: `debug-web/internal/web/handlers_test.go`
- Test: `debug-web/internal/app/app_test.go`

- [ ] **Step 1: Run Python unit tests**

Run: `uv run pytest tests/test_gateway_result_events.py tests/test_gateway_search_breakdown.py -v`  
Expected: PASS

- [ ] **Step 2: Run Go unit tests**

Run: `cd debug-web && go test ./... -v`  
Expected: PASS

- [ ] **Step 3: Run combined verification command**

Run: `uv run pytest tests/test_gateway_result_events.py tests/test_gateway_search_breakdown.py -q && cd debug-web && go test ./...`  
Expected: all tests pass with exit code 0

- [ ] **Step 4: Commit**

```bash
git add tests debug-web README.md compose.yaml src/picca_search
git commit -m "test: verify debug web application and gateway extensions"
```

## Self-Review

- Spec coverage: the plan covers the Go debug web service, sqlite state ownership, S3 upload/proxy, RabbitMQ job and result queues, gateway result event publication, gateway search score breakdowns, Docker/Compose wiring, and unit-test-only verification.
- Placeholder scan: no `TODO`/`TBD` placeholders remain; each task includes concrete files, test commands, and minimal code sketches.
- Type consistency: the plan consistently uses `ImageJobResultMessage`, `IngestionEvent`, `score_breakdown`, `RABBITMQ_RESULT_QUEUE`, and `debug-web` naming throughout.
