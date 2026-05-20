from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class GatewaySettings:
    host: str = "0.0.0.0"
    port: int = 8000
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "picca_images"
    rabbitmq_url: str = "amqp://guest:guest@rabbitmq:5672/%2F"
    rabbitmq_queue: str = "image_jobs"
    rabbitmq_heartbeat: int = 300
    s3_endpoint_url: str = "http://seaweedfs-s3:8333"
    s3_bucket: str = "images"
    s3_access_key_id: str | None = "seaweedfs"
    s3_secret_access_key: str | None = "seaweedfs"
    aws_web_identity_token_file: str | None = None
    aws_endpoint_url_sts: str | None = None
    aws_endpoint_url_s3: str | None = None
    aws_region: str | None = None
    aws_role_arn: str | None = None
    dense_service_url: str = "http://dense-service:8001"
    sparse_service_url: str = "http://sparse-service:8002"
    ocr_service_url: str = "http://ocr-service:8003"
    caption_service_url: str = "http://caption-service:8004"
    batch_size: int = 8
    batch_wait_seconds: float = 2.0

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        aws_web_identity_token_file = _optional_env("AWS_WEB_IDENTITY_TOKEN_FILE")
        aws_role_arn = _optional_env("AWS_ROLE_ARN")
        uses_web_identity = bool(aws_web_identity_token_file or aws_role_arn)
        return cls(
            host=os.getenv("GATEWAY_HOST", "0.0.0.0"),
            port=_parse_port_env("GATEWAY_PORT", 8000),
            qdrant_url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "picca_images"),
            rabbitmq_url=os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/%2F"),
            rabbitmq_queue=os.getenv("RABBITMQ_QUEUE", "image_jobs"),
            rabbitmq_heartbeat=int(os.getenv("GATEWAY_RABBITMQ_HEARTBEAT", "300")),
            s3_endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://seaweedfs-s3:8333"),
            s3_bucket=os.getenv("S3_BUCKET", "images"),
            s3_access_key_id=_optional_env("S3_ACCESS_KEY_ID", default=None if uses_web_identity else "seaweedfs"),
            s3_secret_access_key=_optional_env("S3_SECRET_ACCESS_KEY", default=None if uses_web_identity else "seaweedfs"),
            aws_web_identity_token_file=aws_web_identity_token_file,
            aws_endpoint_url_sts=_optional_env("AWS_ENDPOINT_URL_STS"),
            aws_endpoint_url_s3=_optional_env("AWS_ENDPOINT_URL_S3"),
            aws_region=_optional_env("AWS_REGION"),
            aws_role_arn=aws_role_arn,
            dense_service_url=os.getenv("DENSE_SERVICE_URL", "http://dense-service:8001"),
            sparse_service_url=os.getenv("SPARSE_SERVICE_URL", "http://sparse-service:8002"),
            ocr_service_url=os.getenv("OCR_SERVICE_URL", "http://ocr-service:8003"),
            caption_service_url=os.getenv("CAPTION_SERVICE_URL", "http://caption-service:8004"),
            batch_size=int(os.getenv("INGEST_BATCH_SIZE", "8")),
            batch_wait_seconds=float(os.getenv("INGEST_BATCH_WAIT_SECONDS", "2.0")),
        )

    @property
    def uses_web_identity(self) -> bool:
        return bool(self.aws_web_identity_token_file or self.aws_role_arn)

    @property
    def resolved_s3_endpoint_url(self) -> str:
        return self.aws_endpoint_url_s3 or self.s3_endpoint_url


def _parse_port_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError:
        parsed = urlparse(raw_value)
        if parsed.port is not None:
            return parsed.port
        raise ValueError(f"{name} must be an integer or URL with an explicit port: {raw_value!r}") from None


def _optional_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value
