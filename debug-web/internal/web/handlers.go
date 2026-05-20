package web

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/walnuts1018/picca-ai-prototype/debug-web/internal/config"
	"github.com/walnuts1018/picca-ai-prototype/debug-web/internal/queue"
	"github.com/walnuts1018/picca-ai-prototype/debug-web/internal/s3client"
	"github.com/walnuts1018/picca-ai-prototype/debug-web/internal/search"
	"github.com/walnuts1018/picca-ai-prototype/debug-web/internal/storage"
)

type StatusUpdater interface {
	UpdateStatus(ctx context.Context, imageID string, status string, occurredAt time.Time, errorMessage string) error
}

type UploadPublisher interface {
	PublishJob(queueName string, message queue.JobMessage) error
}

type SearchClient interface {
	Search(ctx context.Context, request search.Request) (search.Response, error)
}

type ObjectStore interface {
	Upload(ctx context.Context, objectKey string, contentType string, body []byte) error
	Download(ctx context.Context, objectKey string) (s3client.Object, error)
}

type Repository interface {
	UpsertQueued(ctx context.Context, record storage.ImageRecord) error
	ListRecent(ctx context.Context, limit int) ([]storage.ImageRecord, error)
	GetByImageID(ctx context.Context, imageID string) (storage.ImageRecord, error)
}

type Handler struct {
	cfg      config.Config
	repo     Repository
	queue    UploadPublisher
	search   SearchClient
	s3       ObjectStore
	template *template.Template
}

type UploadRow struct {
	ImageID      string `json:"image_id"`
	ObjectKey    string `json:"object_key"`
	ImageSrc     string `json:"image_src"`
	Filename     string `json:"filename"`
	Status       string `json:"status"`
	UpdatedAt    string `json:"updated_at"`
	ErrorMessage string `json:"error_message"`
}

type SearchRow struct {
	ImageID        string
	ImageSrc       string
	Score          float64
	Caption        string
	OCRText        string
	ScoreBreakdown *search.ScoreBreakdown
}

type SearchView struct {
	Query   string
	Limit   int
	Results []SearchRow
}

type IndexData struct {
	FlashError               string
	Uploads                  []UploadRow
	Search                   SearchView
	StatusPollInterval       string
	StatusPollIntervalMillis int64
}

func New(cfg config.Config, repo Repository, queue UploadPublisher, searchClient SearchClient, s3 ObjectStore) (*Handler, error) {
	tmpl, err := template.ParseFiles("internal/web/templates/index.html")
	if err != nil {
		return nil, err
	}
	return &Handler{
		cfg:      cfg,
		repo:     repo,
		queue:    queue,
		search:   searchClient,
		s3:       s3,
		template: tmpl,
	}, nil
}

func (h *Handler) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", h.index)
	mux.HandleFunc("/uploads", h.uploads)
	mux.HandleFunc("/uploads/statuses", h.statuses)
	mux.HandleFunc("/search", h.searchHandler)
	mux.HandleFunc("/images/", h.imageProxy)
	mux.HandleFunc("/healthz", h.healthz)
	return mux
}

func (h *Handler) index(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	data, err := h.indexData(r.Context(), "", SearchView{Limit: 10})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	h.renderIndex(w, data)
}

func (h *Handler) uploads(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, h.cfg.MaxUploadSizeBytes)
	if err := r.ParseMultipartForm(h.cfg.MaxUploadSizeBytes); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	files := r.MultipartForm.File["images"]
	if len(files) == 0 {
		http.Error(w, "images is required", http.StatusBadRequest)
		return
	}
	for _, header := range files {
		if err := h.processUpload(r.Context(), header); err != nil {
			data, dataErr := h.indexData(r.Context(), err.Error(), SearchView{Limit: 10})
			if dataErr != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			w.WriteHeader(http.StatusBadRequest)
			h.renderIndex(w, data)
			return
		}
	}
	http.Redirect(w, r, "/", http.StatusSeeOther)
}

func (h *Handler) statuses(w http.ResponseWriter, r *http.Request) {
	rows, err := h.repo.ListRecent(r.Context(), 100)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	payload := make([]UploadRow, 0, len(rows))
	for _, row := range rows {
		payload = append(payload, uploadRow(row))
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(payload)
}

func (h *Handler) searchHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	limit := 10
	if raw := strings.TrimSpace(r.FormValue("limit")); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			limit = parsed
		}
	}
	query := strings.TrimSpace(r.FormValue("query"))
	searchView := SearchView{Query: query, Limit: limit}
	if query != "" {
		response, err := h.search.Search(r.Context(), search.Request{
			Query:       query,
			Limit:       limit,
			Diagnostics: true,
		})
		if err != nil {
			data, dataErr := h.indexData(r.Context(), err.Error(), searchView)
			if dataErr != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			w.WriteHeader(http.StatusBadGateway)
			h.renderIndex(w, data)
			return
		}
		searchView.Results = make([]SearchRow, 0, len(response.Results))
		for _, result := range response.Results {
			objectKey := parseObjectKey(result.Path)
			searchView.Results = append(searchView.Results, SearchRow{
				ImageID:        result.ImageID,
				ImageSrc:       buildImageSrc(result.ImageID, objectKey),
				Score:          result.Score,
				Caption:        result.Caption,
				OCRText:        result.OCRText,
				ScoreBreakdown: result.ScoreBreakdown,
			})
		}
	}
	data, err := h.indexData(r.Context(), "", searchView)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	h.renderIndex(w, data)
}

func (h *Handler) imageProxy(w http.ResponseWriter, r *http.Request) {
	imageID := strings.TrimPrefix(r.URL.Path, "/images/")
	if imageID == "" {
		http.NotFound(w, r)
		return
	}
	record, err := h.repo.GetByImageID(r.Context(), imageID)
	objectKey := r.URL.Query().Get("object_key")
	if err == nil {
		objectKey = record.ObjectKey
	} else if !storage.IsNotFound(err) {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if strings.TrimSpace(objectKey) == "" {
		if storage.IsNotFound(err) {
			http.NotFound(w, r)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	object, err := h.s3.Download(r.Context(), objectKey)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", object.ContentType)
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(object.Body)
}

func (h *Handler) healthz(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}

func (h *Handler) processUpload(ctx context.Context, header *multipart.FileHeader) error {
	file, err := header.Open()
	if err != nil {
		return err
	}
	defer file.Close()

	contentType := header.Header.Get("Content-Type")
	if !strings.HasPrefix(contentType, "image/") {
		if guessed := mimeTypeFromFilename(header.Filename); guessed != "" {
			contentType = guessed
		}
	}
	if !strings.HasPrefix(contentType, "image/") {
		return fmt.Errorf("unsupported content type for %s", header.Filename)
	}
	body, err := io.ReadAll(file)
	if err != nil {
		return err
	}
	objectKey := s3client.BuildObjectKey(h.cfg.UploadObjectPrefix, header.Filename)
	if err := h.s3.Upload(ctx, objectKey, contentType, body); err != nil {
		return err
	}
	now := time.Now().UTC()
	record := storage.ImageRecord{
		ImageID:      objectKey,
		ObjectKey:    objectKey,
		Filename:     header.Filename,
		ContentType:  contentType,
		Status:       "queued",
		ErrorMessage: "",
		UploadedAt:   now,
		UpdatedAt:    now,
	}
	if err := h.repo.UpsertQueued(ctx, record); err != nil {
		return err
	}
	if err := h.queue.PublishJob(h.cfg.RabbitMQJobQueue, queue.JobMessage{ImageID: objectKey}); err != nil {
		return err
	}
	return nil
}

func (h *Handler) renderIndex(w http.ResponseWriter, data IndexData) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_ = h.template.Execute(w, data)
}

func (h *Handler) indexData(ctx context.Context, flashError string, searchView SearchView) (IndexData, error) {
	rows, err := h.repo.ListRecent(ctx, 100)
	if err != nil {
		return IndexData{}, err
	}
	uploads := make([]UploadRow, 0, len(rows))
	for _, row := range rows {
		uploads = append(uploads, uploadRow(row))
	}
	if searchView.Limit == 0 {
		searchView.Limit = 10
	}
	return IndexData{
		FlashError:               flashError,
		Uploads:                  uploads,
		Search:                   searchView,
		StatusPollInterval:       h.cfg.StatusPollInterval.String(),
		StatusPollIntervalMillis: h.cfg.StatusPollInterval.Milliseconds(),
	}, nil
}

func uploadRow(record storage.ImageRecord) UploadRow {
	return UploadRow{
		ImageID:      record.ImageID,
		ObjectKey:    record.ObjectKey,
		ImageSrc:     buildImageSrc(record.ImageID, record.ObjectKey),
		Filename:     record.Filename,
		Status:       record.Status,
		UpdatedAt:    record.UpdatedAt.Local().Format(time.RFC3339),
		ErrorMessage: record.ErrorMessage,
	}
}

func parseObjectKey(path string) string {
	if strings.HasPrefix(path, "s3://") {
		trimmed := strings.TrimPrefix(path, "s3://")
		parts := strings.SplitN(trimmed, "/", 2)
		if len(parts) == 2 {
			return parts[1]
		}
	}
	return ""
}

func buildImageSrc(imageID string, objectKey string) string {
	path := strings.ReplaceAll(url.PathEscape(imageID), "%2F", "/")
	if objectKey == "" || objectKey == imageID {
		return "/images/" + path
	}
	return "/images/" + path + "?object_key=" + url.QueryEscape(objectKey)
}

func mimeTypeFromFilename(filename string) string {
	switch strings.ToLower(filepath.Ext(filename)) {
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".png":
		return "image/png"
	case ".gif":
		return "image/gif"
	case ".webp":
		return "image/webp"
	default:
		return ""
	}
}

func ConsumeResultEvent(ctx context.Context, updater StatusUpdater, message queue.ResultMessage) error {
	occurredAt, err := time.Parse(time.RFC3339, message.OccurredAt)
	if err != nil {
		return err
	}
	err = updater.UpdateStatus(ctx, message.ImageID, message.Status, occurredAt, message.ErrorMessage)
	if err != nil && !errors.Is(err, nil) {
		return err
	}
	return nil
}
