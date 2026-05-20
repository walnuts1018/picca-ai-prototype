package storage

import (
	"context"
	"database/sql"
	"errors"
	"time"

	_ "modernc.org/sqlite"
)

type ImageRecord struct {
	ImageID      string
	ObjectKey    string
	Filename     string
	ContentType  string
	Status       string
	ErrorMessage string
	UploadedAt   time.Time
	UpdatedAt    time.Time
}

type Repository struct {
	db *sql.DB
}

func Open(path string) (*Repository, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	repo := &Repository{db: db}
	if err := repo.init(); err != nil {
		_ = db.Close()
		return nil, err
	}
	return repo, nil
}

func (r *Repository) Close() error {
	return r.db.Close()
}

func (r *Repository) init() error {
	_, err := r.db.Exec(`
CREATE TABLE IF NOT EXISTS image_records (
	image_id TEXT PRIMARY KEY,
	object_key TEXT NOT NULL UNIQUE,
	filename TEXT NOT NULL,
	content_type TEXT NOT NULL,
	status TEXT NOT NULL,
	error_message TEXT NOT NULL DEFAULT '',
	uploaded_at TEXT NOT NULL,
	updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_image_records_updated_at ON image_records(updated_at DESC);
`)
	return err
}

func (r *Repository) UpsertQueued(ctx context.Context, record ImageRecord) error {
	_, err := r.db.ExecContext(ctx, `
INSERT INTO image_records (
	image_id, object_key, filename, content_type, status, error_message, uploaded_at, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(image_id) DO UPDATE SET
	object_key = excluded.object_key,
	filename = excluded.filename,
	content_type = excluded.content_type,
	status = excluded.status,
	error_message = excluded.error_message,
	uploaded_at = excluded.uploaded_at,
	updated_at = excluded.updated_at
`, record.ImageID, record.ObjectKey, record.Filename, record.ContentType, record.Status, record.ErrorMessage, record.UploadedAt.UTC().Format(time.RFC3339), record.UpdatedAt.UTC().Format(time.RFC3339))
	return err
}

func (r *Repository) UpdateStatus(ctx context.Context, imageID string, status string, occurredAt time.Time, errorMessage string) error {
	result, err := r.db.ExecContext(ctx, `
UPDATE image_records
SET status = ?, error_message = ?, updated_at = ?
WHERE image_id = ?
`, status, errorMessage, occurredAt.UTC().Format(time.RFC3339), imageID)
	if err != nil {
		return err
	}
	rows, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if rows == 0 {
		return sql.ErrNoRows
	}
	return nil
}

func (r *Repository) GetByImageID(ctx context.Context, imageID string) (ImageRecord, error) {
	row := r.db.QueryRowContext(ctx, `
SELECT image_id, object_key, filename, content_type, status, error_message, uploaded_at, updated_at
FROM image_records
WHERE image_id = ?
`, imageID)
	return scanRecord(row)
}

func (r *Repository) ListRecent(ctx context.Context, limit int) ([]ImageRecord, error) {
	rows, err := r.db.QueryContext(ctx, `
SELECT image_id, object_key, filename, content_type, status, error_message, uploaded_at, updated_at
FROM image_records
ORDER BY updated_at DESC
LIMIT ?
`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []ImageRecord
	for rows.Next() {
		record, err := scanRows(rows)
		if err != nil {
			return nil, err
		}
		results = append(results, record)
	}
	return results, rows.Err()
}

func scanRecord(scanner interface {
	Scan(dest ...any) error
}) (ImageRecord, error) {
	record, err := scan(scanner)
	if err != nil {
		return ImageRecord{}, err
	}
	return record, nil
}

func scanRows(scanner interface {
	Scan(dest ...any) error
}) (ImageRecord, error) {
	return scan(scanner)
}

func scan(scanner interface {
	Scan(dest ...any) error
}) (ImageRecord, error) {
	var record ImageRecord
	var uploadedAt string
	var updatedAt string
	err := scanner.Scan(
		&record.ImageID,
		&record.ObjectKey,
		&record.Filename,
		&record.ContentType,
		&record.Status,
		&record.ErrorMessage,
		&uploadedAt,
		&updatedAt,
	)
	if err != nil {
		return ImageRecord{}, err
	}
	record.UploadedAt, err = time.Parse(time.RFC3339, uploadedAt)
	if err != nil {
		return ImageRecord{}, err
	}
	record.UpdatedAt, err = time.Parse(time.RFC3339, updatedAt)
	if err != nil {
		return ImageRecord{}, err
	}
	return record, nil
}

func IsNotFound(err error) bool {
	return errors.Is(err, sql.ErrNoRows)
}
