package web

import (
	"bytes"
	"context"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/walnuts1018/picca-ai-prototype/debug-web/internal/config"
	"github.com/walnuts1018/picca-ai-prototype/debug-web/internal/queue"
	"github.com/walnuts1018/picca-ai-prototype/debug-web/internal/s3client"
	"github.com/walnuts1018/picca-ai-prototype/debug-web/internal/search"
	"github.com/walnuts1018/picca-ai-prototype/debug-web/internal/storage"
)

func TestUploadPublishesAndPersists(t *testing.T) {
	handler := newTestHandler(t)

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("images", "a.jpg")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write([]byte("jpegdata")); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(http.MethodPost, "/uploads", &body)
	request.Header.Set("Content-Type", writer.FormDataContentType())
	recorder := httptest.NewRecorder()

	handler.Routes().ServeHTTP(recorder, request)

	if recorder.Code != http.StatusSeeOther {
		t.Fatalf("status = %d body=%s", recorder.Code, recorder.Body.String())
	}
	if len(handler.queue.(*fakeQueue).published) != 1 {
		t.Fatalf("published = %d", len(handler.queue.(*fakeQueue).published))
	}
	rows, err := handler.repo.ListRecent(context.Background(), 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0].Status != "queued" {
		t.Fatalf("rows = %#v", rows)
	}
}

func TestSearchRendersResults(t *testing.T) {
	handler := newTestHandler(t)
	if err := handler.repo.UpsertQueued(context.Background(), storage.ImageRecord{
		ImageID:     "debug/a.jpg",
		ObjectKey:   "debug/a.jpg",
		Filename:    "a.jpg",
		ContentType: "image/jpeg",
		Status:      "indexed",
		UploadedAt:  time.Now().UTC(),
		UpdatedAt:   time.Now().UTC(),
	}); err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(http.MethodPost, "/search", strings.NewReader("query=torii&limit=5"))
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	recorder := httptest.NewRecorder()

	handler.Routes().ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d", recorder.Code)
	}
	if !strings.Contains(recorder.Body.String(), "debug/a.jpg") {
		t.Fatalf("body = %s", recorder.Body.String())
	}
}

func TestImageProxyReturnsStoredImage(t *testing.T) {
	handler := newTestHandler(t)
	if err := handler.repo.UpsertQueued(context.Background(), storage.ImageRecord{
		ImageID:     "debug/a.jpg",
		ObjectKey:   "debug/a.jpg",
		Filename:    "a.jpg",
		ContentType: "image/jpeg",
		Status:      "indexed",
		UploadedAt:  time.Now().UTC(),
		UpdatedAt:   time.Now().UTC(),
	}); err != nil {
		t.Fatal(err)
	}
	handler.s3.(*fakeS3).objects["debug/a.jpg"] = s3client.Object{
		Body:        []byte("jpegdata"),
		ContentType: "image/jpeg",
	}

	request := httptest.NewRequest(http.MethodGet, "/images/debug/a.jpg", nil)
	recorder := httptest.NewRecorder()

	handler.Routes().ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d", recorder.Code)
	}
	if recorder.Header().Get("Content-Type") != "image/jpeg" {
		t.Fatalf("content-type = %q", recorder.Header().Get("Content-Type"))
	}
}

func newTestHandler(t *testing.T) *Handler {
	t.Helper()
	tmpDir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(tmpDir, "internal/web/templates"), 0o755); err != nil {
		t.Fatal(err)
	}
	templatePath := filepath.Join(tmpDir, "internal/web/templates/index.html")
	templateBody, err := os.ReadFile(filepath.Join("templates", "index.html"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(templatePath, templateBody, 0o644); err != nil {
		t.Fatal(err)
	}
	previousWD, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(tmpDir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(previousWD)
	})

	repo, err := storage.Open("file::memory:?cache=shared")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = repo.Close()
	})

	handler, err := New(config.Config{
		RabbitMQJobQueue:   "image_jobs",
		UploadObjectPrefix: "debug/",
		StatusPollInterval: 3 * time.Second,
		MaxUploadSizeBytes: 64 * 1024 * 1024,
	}, repo, &fakeQueue{}, fakeSearchClient{}, &fakeS3{objects: map[string]s3client.Object{}})
	if err != nil {
		t.Fatal(err)
	}
	return handler
}

type fakeQueue struct {
	published []queue.JobMessage
}

func (f *fakeQueue) PublishJob(_ string, message queue.JobMessage) error {
	f.published = append(f.published, message)
	return nil
}

type fakeS3 struct {
	objects map[string]s3client.Object
}

func (f *fakeS3) Upload(_ context.Context, objectKey string, contentType string, body []byte) error {
	f.objects[objectKey] = s3client.Object{Body: append([]byte(nil), body...), ContentType: contentType}
	return nil
}

func (f *fakeS3) Download(_ context.Context, objectKey string) (s3client.Object, error) {
	object, ok := f.objects[objectKey]
	if !ok {
		return s3client.Object{}, os.ErrNotExist
	}
	return object, nil
}

type fakeSearchClient struct{}

func (fakeSearchClient) Search(_ context.Context, request search.Request) (search.Response, error) {
	if request.Query == "" {
		return search.Response{}, nil
	}
	return search.Response{
		Results: []search.Result{{
			ImageID: "debug/a.jpg",
			Score:   1.5,
			Caption: "red gate",
			OCRText: "torii",
			ScoreBreakdown: &search.ScoreBreakdown{
				DenseScore: 1.0,
			},
		}},
	}, nil
}

type updatingRepo struct {
	updateFn func(ctx context.Context, imageID string, status string, occurredAt time.Time, errorMessage string) error
}

func (u updatingRepo) UpdateStatus(ctx context.Context, imageID string, status string, occurredAt time.Time, errorMessage string) error {
	return u.updateFn(ctx, imageID, status, occurredAt, errorMessage)
}

func TestConsumeResultEvent(t *testing.T) {
	called := false
	err := ConsumeResultEvent(context.Background(), updatingRepo{
		updateFn: func(_ context.Context, imageID string, status string, occurredAt time.Time, errorMessage string) error {
			called = true
			if imageID != "debug/a.jpg" || status != "indexed" || !occurredAt.Equal(time.Date(2026, 5, 21, 1, 2, 3, 0, time.UTC)) || errorMessage != "" {
				t.Fatalf("unexpected update: %s %s %s %q", imageID, status, occurredAt, errorMessage)
			}
			return nil
		},
	}, queue.ResultMessage{ImageID: "debug/a.jpg", Status: "indexed", OccurredAt: "2026-05-21T01:02:03Z"})
	if err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("update not called")
	}
}

func TestTemplateFixtureReadable(t *testing.T) {
	file, err := os.Open(filepath.Join("templates", "index.html"))
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	if _, err := io.ReadAll(file); err != nil {
		t.Fatal(err)
	}
}
