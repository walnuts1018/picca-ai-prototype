package s3client

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/url"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"

	cfgpkg "github.com/walnuts1018/picca-ai-prototype/debug-web/internal/config"
)

type Client struct {
	api    *s3.Client
	bucket string
}

type Object struct {
	Body        []byte
	ContentType string
}

func New(ctx context.Context, cfg cfgpkg.Config) (*Client, error) {
	loadOptions := []func(*awsconfig.LoadOptions) error{}
	if cfg.AWSRegion != "" {
		loadOptions = append(loadOptions, awsconfig.WithRegion(cfg.AWSRegion))
	}
	if !cfg.UsesWebIdentity() && cfg.S3AccessKeyID != "" && cfg.S3SecretAccessKey != "" {
		loadOptions = append(loadOptions, awsconfig.WithCredentialsProvider(
			credentials.NewStaticCredentialsProvider(cfg.S3AccessKeyID, cfg.S3SecretAccessKey, ""),
		))
	}
	awsCfg, err := awsconfig.LoadDefaultConfig(ctx, loadOptions...)
	if err != nil {
		return nil, err
	}
	endpointURL := cfg.ResolvedS3EndpointURL()
	options := func(o *s3.Options) {
		o.UsePathStyle = true
		if endpointURL != "" {
			o.BaseEndpoint = aws.String(endpointURL)
		}
	}
	return &Client{
		api:    s3.NewFromConfig(awsCfg, options),
		bucket: cfg.S3Bucket,
	}, nil
}

func (c *Client) Upload(ctx context.Context, objectKey string, contentType string, body []byte) error {
	if _, err := c.api.PutObject(ctx, &s3.PutObjectInput{
		Bucket:      aws.String(c.bucket),
		Key:         aws.String(objectKey),
		Body:        bytes.NewReader(body),
		ContentType: aws.String(contentType),
	}); err != nil {
		return err
	}
	return nil
}

func (c *Client) Download(ctx context.Context, objectKey string) (Object, error) {
	output, err := c.api.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(c.bucket),
		Key:    aws.String(objectKey),
	})
	if err != nil {
		return Object{}, err
	}
	defer output.Body.Close()
	body, err := io.ReadAll(output.Body)
	if err != nil {
		return Object{}, err
	}
	contentType := "application/octet-stream"
	if output.ContentType != nil && strings.TrimSpace(*output.ContentType) != "" {
		contentType = *output.ContentType
	}
	return Object{Body: body, ContentType: contentType}, nil
}

func BuildObjectKey(prefix string, filename string) string {
	filename = strings.ReplaceAll(filename, "\\", "_")
	filename = strings.ReplaceAll(filename, "/", "_")
	return fmt.Sprintf("%s%s", prefix, url.PathEscape(filename))
}
