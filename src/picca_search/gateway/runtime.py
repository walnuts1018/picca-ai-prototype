from __future__ import annotations

import threading
import time

import boto3
from qdrant_client import QdrantClient

from picca_search.gateway.config import GatewaySettings
from picca_search.gateway.ingestion import GatewayIngestionService, PendingImageJob
from picca_search.gateway.search_api import GatewaySearchDependencies, create_search_app
from picca_search.infrastructure.model_client import (
    CaptionModelClient,
    DenseModelClient,
    OcrModelClient,
    SparseModelClient,
)
from picca_search.infrastructure.object_storage import SeaweedObjectStorage
from picca_search.infrastructure.qdrant_index import QdrantImageIndex
from picca_search.infrastructure.rabbitmq_queue import RabbitMqImageJobQueue


def create_gateway_app(settings: GatewaySettings):
    dense_client = DenseModelClient(settings.dense_service_url)
    sparse_client = SparseModelClient(settings.sparse_service_url)
    search_dependencies = GatewaySearchDependencies(
        dense_client=dense_client,
        sparse_client=sparse_client,
        index=QdrantImageIndex(QdrantClient(url=settings.qdrant_url), settings.qdrant_collection),
    )
    app = create_search_app(search_dependencies)

    @app.on_event("startup")
    def start_consumer() -> None:
        app.state.consumer_stop = threading.Event()
        app.state.consumer_thread = threading.Thread(
            target=_run_consumer_loop,
            args=(settings, app.state.consumer_stop),
            daemon=True,
        )
        app.state.consumer_thread.start()

    @app.on_event("shutdown")
    def stop_consumer() -> None:
        stop_event = getattr(app.state, "consumer_stop", None)
        thread = getattr(app.state, "consumer_thread", None)
        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            thread.join(timeout=10)

    return app


def _run_consumer_loop(settings: GatewaySettings, stop_event: threading.Event) -> None:
    s3_client = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
    )
    storage = SeaweedObjectStorage(s3_client=s3_client, bucket=settings.s3_bucket)
    queue = RabbitMqImageJobQueue(settings.rabbitmq_url, settings.rabbitmq_queue)
    ingestion = GatewayIngestionService(
        storage=storage,
        dense_client=DenseModelClient(settings.dense_service_url),
        sparse_client=SparseModelClient(settings.sparse_service_url),
        ocr_client=OcrModelClient(settings.ocr_service_url),
        caption_client=CaptionModelClient(settings.caption_service_url),
        index=QdrantImageIndex(QdrantClient(url=settings.qdrant_url), settings.qdrant_collection),
    )
    try:
        while not stop_event.is_set():
            deliveries = queue.get_batch(settings.batch_size, settings.batch_wait_seconds)
            if not deliveries:
                time.sleep(0.2)
                continue
            outcome = ingestion.process_jobs(
                [PendingImageJob(delivery_tag=item.delivery_tag, image_id=item.message.image_id) for item in deliveries]
            )
            for delivery_tag in outcome.acked_delivery_tags:
                queue.ack(delivery_tag)
            for delivery_tag in outcome.requeue_delivery_tags:
                queue.nack(delivery_tag, requeue=True)
            for delivery_tag in outcome.dead_letter_delivery_tags:
                queue.nack(delivery_tag, requeue=False)
    finally:
        queue.close()
