package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"

	ocispecv1 "github.com/opencontainers/image-spec/specs-go/v1"
	oras "oras.land/oras-go/v2"
	"oras.land/oras-go/v2/content"
	"oras.land/oras-go/v2/content/file"
	"oras.land/oras-go/v2/registry/remote"
	"oras.land/oras-go/v2/registry/remote/auth"
	"oras.land/oras-go/v2/registry/remote/retry"
)

const defaultArtifactType = "application/vnd.picca.models.v1"

var defaultPlatforms = []ocispecv1.Platform{
	{
		Architecture: "amd64",
		OS:           "linux",
	},
}

type repositoryCredential struct {
	HostName string
	Username string
	Password string
}

func init() {
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, nil)))
}

func main() {
	var dir string
	var tag string
	var repository string
	var username string
	var password string
	var artifactType string

	flag.StringVar(&dir, "dir", "", "Directory containing model files to push")
	flag.StringVar(&tag, "tag", "latest", "Tag for the artifact")
	flag.StringVar(&repository, "repo", "", "Repository to push to (e.g. ghcr.io/user/repo)")
	flag.StringVar(&username, "username", firstNonEmpty(os.Getenv("OCI_REGISTRY_USERNAME"), os.Getenv("GITHUB_ACTOR")), "Registry username")
	flag.StringVar(&password, "password", firstNonEmpty(os.Getenv("OCI_REGISTRY_PASSWORD"), os.Getenv("GITHUB_TOKEN")), "Registry password or token")
	flag.StringVar(&artifactType, "artifact-type", defaultArtifactType, "Artifact type stored in manifest config")
	flag.Parse()

	if dir == "" || repository == "" {
		flag.Usage()
		os.Exit(1)
	}

	ctx := context.Background()

	host, err := repositoryHost(repository)
	if err != nil {
		slog.Error("invalid repository", "repository", repository, "error", err)
		os.Exit(1)
	}

	credential := repositoryCredential{
		HostName: host,
		Username: username,
		Password: password,
	}
	if err := validateCredential(credential); err != nil {
		slog.Error("invalid registry credential", "error", err)
		os.Exit(1)
	}

	if err := pushOCIArtifact(ctx, dir, repository, tag, artifactType, credential); err != nil {
		slog.Error("failed to push OCI artifact", "error", err)
		os.Exit(1)
	}

	slog.Info("successfully pushed OCI artifact", "repository", repository, "tag", tag, "dir", dir)
}

func pushOCIArtifact(
	ctx context.Context,
	sourceDir string,
	repository string,
	tag string,
	artifactType string,
	credential repositoryCredential,
) error {
	stagingDir, err := os.MkdirTemp("", "push-models-oci-*")
	if err != nil {
		return fmt.Errorf("failed to create staging dir: %w", err)
	}
	defer os.RemoveAll(stagingDir)

	artifactPaths, err := stageArtifacts(sourceDir, stagingDir)
	if err != nil {
		return err
	}

	store, err := file.New(stagingDir)
	if err != nil {
		return fmt.Errorf("failed to create file store: %w", err)
	}
	defer store.Close()

	layerDescriptors, err := addLayers(ctx, store, stagingDir, artifactPaths)
	if err != nil {
		return err
	}

	configDescriptor, err := pushImageConfig(ctx, store, artifactType, defaultPlatforms[0])
	if err != nil {
		return err
	}

	manifestDescriptor, err := oras.PackManifest(
		ctx,
		store,
		oras.PackManifestVersion1_1,
		ocispecv1.MediaTypeImageManifest,
		oras.PackManifestOptions{
			Layers:           layerDescriptors,
			ConfigDescriptor: &configDescriptor,
		},
	)
	if err != nil {
		return fmt.Errorf("failed to pack manifest: %w", err)
	}

	if err := store.Tag(ctx, manifestDescriptor, tag); err != nil {
		return fmt.Errorf("failed to tag manifest: %w", err)
	}

	repo, err := remote.NewRepository(repository)
	if err != nil {
		return fmt.Errorf("failed to create remote repository: %w", err)
	}

	repo.Client = &auth.Client{
		Client: retry.DefaultClient,
		Cache:  auth.NewCache(),
		Credential: auth.StaticCredential(credential.HostName, auth.Credential{
			Username: credential.Username,
			Password: credential.Password,
		}),
	}

	if _, err := oras.Copy(ctx, store, tag, repo, tag, oras.DefaultCopyOptions); err != nil {
		return fmt.Errorf("failed to push OCI artifact: %w", err)
	}

	return nil
}

func addLayers(ctx context.Context, store *file.Store, stagingDir string, artifactPaths []string) ([]ocispecv1.Descriptor, error) {
	descriptors := make([]ocispecv1.Descriptor, 0, len(artifactPaths))

	for _, artifactPath := range artifactPaths {
		mediaType := ocispecv1.MediaTypeImageLayer
		if strings.HasSuffix(artifactPath, ".tar.gz") {
			mediaType = ocispecv1.MediaTypeImageLayerGzip
		}

		name, err := filepath.Rel(stagingDir, artifactPath)
		if err != nil {
			return nil, fmt.Errorf("failed to resolve artifact path %s: %w", artifactPath, err)
		}

		descriptor, err := store.Add(ctx, name, mediaType, "")
		if err != nil {
			return nil, fmt.Errorf("failed to add layer %s: %w", name, err)
		}
		descriptors = append(descriptors, descriptor)
	}

	return descriptors, nil
}

func pushImageConfig(
	ctx context.Context,
	store *file.Store,
	artifactType string,
	platform ocispecv1.Platform,
) (ocispecv1.Descriptor, error) {
	configData, err := json.Marshal(ocispecv1.Image{
		Platform: platform,
		Config: ocispecv1.ImageConfig{
			Labels: map[string]string{
				"org.opencontainers.artifact.type": artifactType,
			},
		},
	})
	if err != nil {
		return ocispecv1.Descriptor{}, fmt.Errorf("failed to marshal config: %w", err)
	}

	configDescriptor := content.NewDescriptorFromBytes(ocispecv1.MediaTypeImageConfig, configData)
	if err := store.Push(ctx, configDescriptor, bytes.NewReader(configData)); err != nil {
		return ocispecv1.Descriptor{}, fmt.Errorf("failed to store config: %w", err)
	}

	return configDescriptor, nil
}

func stageArtifacts(sourceDir string, stagingDir string) ([]string, error) {
	entryNames, err := collectTopLevelEntries(sourceDir)
	if err != nil {
		return nil, err
	}
	if len(entryNames) == 0 {
		return nil, fmt.Errorf("no model entries found in %s", sourceDir)
	}

	artifactPaths := make([]string, 0, len(entryNames))
	for _, entryName := range entryNames {
		sourcePath := filepath.Join(sourceDir, entryName)
		info, err := os.Stat(sourcePath)
		if err != nil {
			return nil, fmt.Errorf("failed to stat %s: %w", sourcePath, err)
		}

		if info.IsDir() {
			artifactPath := filepath.Join(stagingDir, entryName+".tar.gz")
			if err := writeTarGz(sourceDir, entryName, artifactPath); err != nil {
				return nil, fmt.Errorf("failed to stage directory %s: %w", entryName, err)
			}
			artifactPaths = append(artifactPaths, artifactPath)
			continue
		}

		artifactPath := filepath.Join(stagingDir, entryName)
		if err := copyFile(sourcePath, artifactPath, info.Mode()); err != nil {
			return nil, fmt.Errorf("failed to stage file %s: %w", entryName, err)
		}
		artifactPaths = append(artifactPaths, artifactPath)
	}

	return artifactPaths, nil
}

func collectTopLevelEntries(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("failed to read dir %s: %w", dir, err)
	}

	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), ".") {
			continue
		}
		names = append(names, entry.Name())
	}

	slices.Sort(names)
	return names, nil
}

func writeTarGz(baseDir string, entryName string, destinationPath string) error {
	file, err := os.Create(destinationPath)
	if err != nil {
		return err
	}
	defer file.Close()

	gzipWriter := gzip.NewWriter(file)
	defer gzipWriter.Close()

	tarWriter := tar.NewWriter(gzipWriter)
	defer tarWriter.Close()

	return filepath.Walk(filepath.Join(baseDir, entryName), func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}

		relPath, err := filepath.Rel(baseDir, path)
		if err != nil {
			return err
		}
		relPath = filepath.ToSlash(relPath)

		var linkTarget string
		if info.Mode()&os.ModeSymlink != 0 {
			linkTarget, err = os.Readlink(path)
			if err != nil {
				return fmt.Errorf("failed to read symlink %s: %w", path, err)
			}
		}

		header, err := tar.FileInfoHeader(info, linkTarget)
		if err != nil {
			return err
		}
		header.Name = relPath

		if err := tarWriter.WriteHeader(header); err != nil {
			return err
		}

		if !info.Mode().IsRegular() {
			return nil
		}

		sourceFile, err := os.Open(path)
		if err != nil {
			return err
		}
		defer sourceFile.Close()

		if _, err := io.Copy(tarWriter, sourceFile); err != nil {
			return err
		}
		return nil
	})
}

func copyFile(sourcePath string, destinationPath string, mode os.FileMode) error {
	sourceFile, err := os.Open(sourcePath)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	destinationFile, err := os.OpenFile(destinationPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode.Perm())
	if err != nil {
		return err
	}
	defer destinationFile.Close()

	if _, err := io.Copy(destinationFile, sourceFile); err != nil {
		return err
	}
	return nil
}

func repositoryHost(repository string) (string, error) {
	parts := strings.SplitN(repository, "/", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" || !strings.Contains(parts[0], ".") {
		return "", fmt.Errorf("repository must include registry host: %s", repository)
	}
	return parts[0], nil
}

func validateCredential(credential repositoryCredential) error {
	if credential.Username == "" || credential.Password == "" {
		return fmt.Errorf("registry credential is required")
	}
	return nil
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
