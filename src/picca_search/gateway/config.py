from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GatewaySettings:
    host: str = "0.0.0.0"
    port: int = 8000
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "picca_images"
    rabbitmq_url: str = "amqp://guest:guest@rabbitmq:5672/%2F"
    rabbitmq_queue: str = "image_jobs"
    s3_endpoint_url: str = "http://seaweedfs-s3:8333"
    s3_bucket: str = "images"
    s3_access_key_id: str = "seaweedfs"
    s3_secret_access_key: str = "seaweedfs"
    dense_service_url: str = "http://dense-service:8001"
    sparse_service_url: str = "http://sparse-service:8002"
    ocr_service_url: str = "http://ocr-service:8003"
    caption_service_url: str = "http://caption-service:8004"
    batch_size: int = 8
    batch_wait_seconds: float = 2.0

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        return cls(
            host=os.getenv("GATEWAY_HOST", "0.0.0.0"),
            port=int(os.getenv("GATEWAY_PORT", "8000")),
            qdrant_url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "picca_images"),
            rabbitmq_url=os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/%2F"),
            rabbitmq_queue=os.getenv("RABBITMQ_QUEUE", "image_jobs"),
            s3_endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://seaweedfs-s3:8333"),
            s3_bucket=os.getenv("S3_BUCKET", "images"),
            s3_access_key_id=os.getenv("S3_ACCESS_KEY_ID", "seaweedfs"),
            s3_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY", "seaweedfs"),
            dense_service_url=os.getenv("DENSE_SERVICE_URL", "http://dense-service:8001"),
            sparse_service_url=os.getenv("SPARSE_SERVICE_URL", "http://sparse-service:8002"),
            ocr_service_url=os.getenv("OCR_SERVICE_URL", "http://ocr-service:8003"),
            caption_service_url=os.getenv("CAPTION_SERVICE_URL", "http://caption-service:8004"),
            batch_size=int(os.getenv("INGEST_BATCH_SIZE", "8")),
            batch_wait_seconds=float(os.getenv("INGEST_BATCH_WAIT_SECONDS", "2.0")),
        )
