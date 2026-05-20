package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

type Config struct {
	Host                    string
	Port                    int
	SQLitePath              string
	GatewayBaseURL          string
	SearchTimeout           time.Duration
	StatusPollInterval      time.Duration
	UploadObjectPrefix      string
	RabbitMQURL             string
	RabbitMQJobQueue        string
	RabbitMQResultQueue     string
	RabbitMQHeartbeat       time.Duration
	S3Bucket                string
	S3EndpointURL           string
	S3AccessKeyID           string
	S3SecretAccessKey       string
	AWSWebIdentityTokenFile string
	AWSEndpointURLSTS       string
	AWSEndpointURLS3        string
	AWSRegion               string
	AWSRoleARN              string
	MaxUploadSizeBytes      int64
}

func Load() (Config, error) {
	searchTimeoutSeconds, err := intEnv("SEARCH_TIMEOUT_SECONDS", 30)
	if err != nil {
		return Config{}, err
	}
	statusPollIntervalMillis, err := intEnv("STATUS_POLL_INTERVAL_MS", 3000)
	if err != nil {
		return Config{}, err
	}
	port, err := intEnv("DEBUG_WEB_PORT", 8080)
	if err != nil {
		return Config{}, err
	}
	maxUploadMegabytes, err := intEnv("MAX_UPLOAD_SIZE_MB", 64)
	if err != nil {
		return Config{}, err
	}
	heartbeatSeconds, err := intEnv("DEBUG_WEB_RABBITMQ_HEARTBEAT", 300)
	if err != nil {
		return Config{}, err
	}

	cfg := Config{
		Host:                    env("DEBUG_WEB_HOST", "0.0.0.0"),
		Port:                    port,
		SQLitePath:              env("SQLITE_PATH", "/data/debug-web.sqlite"),
		GatewayBaseURL:          strings.TrimRight(env("GATEWAY_BASE_URL", "http://gateway:8000"), "/"),
		SearchTimeout:           time.Duration(searchTimeoutSeconds) * time.Second,
		StatusPollInterval:      time.Duration(statusPollIntervalMillis) * time.Millisecond,
		UploadObjectPrefix:      env("UPLOAD_OBJECT_PREFIX", "debug/"),
		RabbitMQURL:             env("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/%2F"),
		RabbitMQJobQueue:        env("RABBITMQ_QUEUE", "image_jobs"),
		RabbitMQResultQueue:     env("RABBITMQ_RESULT_QUEUE", "image_job_results"),
		RabbitMQHeartbeat:       time.Duration(heartbeatSeconds) * time.Second,
		S3Bucket:                env("S3_BUCKET", "images"),
		S3EndpointURL:           env("S3_ENDPOINT_URL", "http://seaweedfs:8333"),
		S3AccessKeyID:           os.Getenv("S3_ACCESS_KEY_ID"),
		S3SecretAccessKey:       os.Getenv("S3_SECRET_ACCESS_KEY"),
		AWSWebIdentityTokenFile: os.Getenv("AWS_WEB_IDENTITY_TOKEN_FILE"),
		AWSEndpointURLSTS:       os.Getenv("AWS_ENDPOINT_URL_STS"),
		AWSEndpointURLS3:        os.Getenv("AWS_ENDPOINT_URL_S3"),
		AWSRegion:               os.Getenv("AWS_REGION"),
		AWSRoleARN:              os.Getenv("AWS_ROLE_ARN"),
		MaxUploadSizeBytes:      int64(maxUploadMegabytes) * 1024 * 1024,
	}
	if cfg.GatewayBaseURL == "" {
		return Config{}, fmt.Errorf("GATEWAY_BASE_URL must not be empty")
	}
	if cfg.UploadObjectPrefix != "" && !strings.HasSuffix(cfg.UploadObjectPrefix, "/") {
		cfg.UploadObjectPrefix += "/"
	}
	return cfg, nil
}

func (c Config) Address() string {
	return fmt.Sprintf("%s:%d", c.Host, c.Port)
}

func (c Config) ResolvedS3EndpointURL() string {
	if c.AWSEndpointURLS3 != "" {
		return c.AWSEndpointURLS3
	}
	return c.S3EndpointURL
}

func (c Config) UsesWebIdentity() bool {
	return c.AWSWebIdentityTokenFile != "" || c.AWSRoleARN != ""
}

func env(name string, fallback string) string {
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	return value
}

func intEnv(name string, fallback int) (int, error) {
	value := os.Getenv(name)
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return 0, fmt.Errorf("%s must be an integer: %w", name, err)
	}
	return parsed, nil
}
