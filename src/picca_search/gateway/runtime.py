from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone

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
from picca_search.infrastructure.rabbitmq_queue import ImageJobResultMessage, RabbitMqImageJobQueue

logger = logging.getLogger(__name__)


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
    try:
        s3_client = _create_s3_client(settings)
        storage = SeaweedObjectStorage(s3_client=s3_client, bucket=settings.s3_bucket)
        queue = RabbitMqImageJobQueue(settings.rabbitmq_url, settings.rabbitmq_queue, heartbeat=settings.rabbitmq_heartbeat)
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

                logger.info(f"Processing batch of {len(deliveries)} jobs")
                for item in deliveries:
                    queue.publish_result(
                        settings.rabbitmq_result_queue,
                        ImageJobResultMessage(
                            image_id=item.message.image_id,
                            status="processing",
                            occurred_at=_utc_now_isoformat(),
                        ),
                    )
                outcome = ingestion.process_jobs(
                    [PendingImageJob(delivery_tag=item.delivery_tag, image_id=item.message.image_id) for item in deliveries]
                )
                image_ids_by_tag = {item.delivery_tag: item.message.image_id for item in deliveries}
                acked_tags = set(outcome.acked_delivery_tags)
                requeue_tags = set(outcome.requeue_delivery_tags)
                dead_letter_tags = set(outcome.dead_letter_delivery_tags)
                handled_tags: set[int] = set()

                if outcome.acked_delivery_tags:
                    logger.info(f"Acking {len(outcome.acked_delivery_tags)} jobs")
                if outcome.requeue_delivery_tags:
                    logger.warning(f"Requeueing {len(outcome.requeue_delivery_tags)} jobs")
                if outcome.dead_letter_delivery_tags:
                    logger.error(f"Dead-lettering {len(outcome.dead_letter_delivery_tags)} jobs")

                for event in outcome.image_result_events:
                    queue.publish_result(
                        settings.rabbitmq_result_queue,
                        ImageJobResultMessage(
                            image_id=event.image_id,
                            status=event.status,
                            occurred_at=_utc_now_isoformat(),
                            error_message=event.error_message,
                        ),
                    )
                    _apply_delivery_disposition(
                        queue=queue,
                        delivery_tag=event.delivery_tag,
                        acked_tags=acked_tags,
                        requeue_tags=requeue_tags,
                        dead_letter_tags=dead_letter_tags,
                    )
                    handled_tags.add(event.delivery_tag)

                for delivery_tag in outcome.acked_delivery_tags:
                    if delivery_tag not in handled_tags:
                        queue.ack(delivery_tag)

                for delivery_tag in outcome.requeue_delivery_tags:
                    if delivery_tag not in handled_tags:
                        queue.nack(delivery_tag, requeue=True)

                for delivery_tag in outcome.dead_letter_delivery_tags:
                    if delivery_tag not in handled_tags:
                        queue.publish_result(
                            settings.rabbitmq_result_queue,
                            ImageJobResultMessage(
                                image_id=image_ids_by_tag[delivery_tag],
                                status="failed",
                                occurred_at=_utc_now_isoformat(),
                                error_message="dead-lettered without explicit ingestion failure event",
                            ),
                        )
                        queue.nack(delivery_tag, requeue=False)
        finally:
            queue.close()
    except Exception:
        logger.exception("Unrecoverable error in consumer thread. Exiting process.")
        os._exit(1)


def _create_s3_client(settings: GatewaySettings):
    session = boto3.session.Session(region_name=settings.aws_region)
    client_kwargs: dict[str, str] = {
        "endpoint_url": settings.resolved_s3_endpoint_url,
    }
    if settings.aws_region is not None:
        client_kwargs["region_name"] = settings.aws_region
    if settings.s3_access_key_id is not None and settings.s3_secret_access_key is not None:
        client_kwargs["aws_access_key_id"] = settings.s3_access_key_id
        client_kwargs["aws_secret_access_key"] = settings.s3_secret_access_key
    return session.client("s3", **client_kwargs)


def _utc_now_isoformat() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _apply_delivery_disposition(
    *,
    queue: RabbitMqImageJobQueue,
    delivery_tag: int,
    acked_tags: set[int],
    requeue_tags: set[int],
    dead_letter_tags: set[int],
) -> None:
    if delivery_tag in acked_tags:
        queue.ack(delivery_tag)
        return
    if delivery_tag in requeue_tags:
        queue.nack(delivery_tag, requeue=True)
        return
    if delivery_tag in dead_letter_tags:
        queue.nack(delivery_tag, requeue=False)
        return
    raise ValueError(f"delivery tag {delivery_tag} has no disposition")
