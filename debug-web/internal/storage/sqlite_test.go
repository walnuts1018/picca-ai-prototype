package storage

import (
	"context"
	"testing"
	"time"
)

func TestRepositoryUpsertQueuedAndUpdateStatus(t *testing.T) {
	repo, err := Open("file::memory:?cache=shared")
	if err != nil {
		t.Fatal(err)
	}
	defer repo.Close()

	now := time.Date(2026, 5, 21, 1, 2, 3, 0, time.UTC)
	record := ImageRecord{
		ImageID:     "debug/a.jpg",
		ObjectKey:   "debug/a.jpg",
		Filename:    "a.jpg",
		ContentType: "image/jpeg",
		Status:      "queued",
		UploadedAt:  now,
		UpdatedAt:   now,
	}
	if err := repo.UpsertQueued(context.Background(), record); err != nil {
		t.Fatal(err)
	}

	if err := repo.UpdateStatus(context.Background(), "debug/a.jpg", "indexed", now.Add(time.Minute), ""); err != nil {
		t.Fatal(err)
	}

	got, err := repo.GetByImageID(context.Background(), "debug/a.jpg")
	if err != nil {
		t.Fatal(err)
	}
	if got.Status != "indexed" {
		t.Fatalf("status = %q", got.Status)
	}
	if got.Filename != "a.jpg" {
		t.Fatalf("filename = %q", got.Filename)
	}
}

func TestRepositoryListRecentOrdersNewestFirst(t *testing.T) {
	repo, err := Open("file::memory:?cache=shared")
	if err != nil {
		t.Fatal(err)
	}
	defer repo.Close()

	base := time.Date(2026, 5, 21, 1, 0, 0, 0, time.UTC)
	for i, imageID := range []string{"debug/a.jpg", "debug/b.jpg"} {
		now := base.Add(time.Duration(i) * time.Minute)
		if err := repo.UpsertQueued(context.Background(), ImageRecord{
			ImageID:     imageID,
			ObjectKey:   imageID,
			Filename:    imageID,
			ContentType: "image/jpeg",
			Status:      "queued",
			UploadedAt:  now,
			UpdatedAt:   now,
		}); err != nil {
			t.Fatal(err)
		}
	}

	got, err := repo.ListRecent(context.Background(), 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("len = %d", len(got))
	}
	if got[0].ImageID != "debug/b.jpg" {
		t.Fatalf("first image = %q", got[0].ImageID)
	}
}
