package main

import (
	"archive/tar"
	"compress/gzip"
	"io"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestDefaultPlatformsContainsOnlyLinuxAmd64(t *testing.T) {
	if len(defaultPlatforms) != 1 {
		t.Fatalf("unexpected platform count: got %d want 1", len(defaultPlatforms))
	}

	platform := defaultPlatforms[0]
	if platform.OS != "linux" || platform.Architecture != "amd64" {
		t.Fatalf("unexpected platform: got %s/%s want linux/amd64", platform.OS, platform.Architecture)
	}
}

func TestStageArtifactsCreatesTarGzForDirectories(t *testing.T) {
	sourceDir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(sourceDir, "model-a", "nested"), 0o755); err != nil {
		t.Fatalf("failed to create model dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(sourceDir, "model-a", "nested", "weights.bin"), []byte("weights"), 0o644); err != nil {
		t.Fatalf("failed to write nested file: %v", err)
	}
	if err := os.WriteFile(filepath.Join(sourceDir, "tokenizer.json"), []byte("{}"), 0o644); err != nil {
		t.Fatalf("failed to write file: %v", err)
	}

	stagingDir := t.TempDir()
	artifacts, err := stageArtifacts(sourceDir, stagingDir)
	if err != nil {
		t.Fatalf("stageArtifacts returned error: %v", err)
	}

	expectedPaths := []string{
		filepath.Join(stagingDir, "model-a.tar.gz"),
		filepath.Join(stagingDir, "tokenizer.json"),
	}
	if !slices.Equal(artifacts, expectedPaths) {
		t.Fatalf("unexpected artifacts: got %v want %v", artifacts, expectedPaths)
	}

	tarNames := readTarGzEntryNames(t, filepath.Join(stagingDir, "model-a.tar.gz"))
	expectedTarNames := []string{"model-a", "model-a/nested", "model-a/nested/weights.bin"}
	if !slices.Equal(tarNames, expectedTarNames) {
		t.Fatalf("unexpected tar entries: got %v want %v", tarNames, expectedTarNames)
	}
}

func readTarGzEntryNames(t *testing.T, path string) []string {
	t.Helper()

	file, err := os.Open(path)
	if err != nil {
		t.Fatalf("failed to open tar.gz: %v", err)
	}
	defer file.Close()

	gzipReader, err := gzip.NewReader(file)
	if err != nil {
		t.Fatalf("failed to create gzip reader: %v", err)
	}
	defer gzipReader.Close()

	tarReader := tar.NewReader(gzipReader)
	var names []string
	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			return names
		}
		if err != nil {
			t.Fatalf("failed to read tar entry: %v", err)
		}
		names = append(names, header.Name)
	}
}
