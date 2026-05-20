from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime

import pika


_IMAGE_JOB_RESULT_STATUSES = frozenset({"processing", "indexed", "failed"})


@dataclass(frozen=True)
class ImageJobMessage:
    image_id: str

    @classmethod
    def from_body(cls, body: bytes) -> "ImageJobMessage":
        payload = json.loads(body.decode("utf-8"))
        image_id = str(payload["image_id"]).strip()
        if image_id == "":
            raise ValueError("image_id must not be blank")
        return cls(image_id=image_id)

    def to_body(self) -> bytes:
        return json.dumps({"image_id": self.image_id}, ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class ImageJobResultMessage:
    image_id: str
    status: str
    occurred_at: str
    error_message: str | None = None

    @classmethod
    def from_body(cls, body: bytes) -> "ImageJobResultMessage":
        payload = json.loads(body.decode("utf-8"))
        image_id = _require_non_blank_string(payload, "image_id")
        status = _require_non_blank_string(payload, "status")
        occurred_at = _require_non_blank_string(payload, "occurred_at")
        error_message = payload.get("error_message")
        if status not in _IMAGE_JOB_RESULT_STATUSES:
            raise ValueError(f"status must be one of {sorted(_IMAGE_JOB_RESULT_STATUSES)}")
        _parse_timestamp(occurred_at)
        normalized_error_message = None if error_message in (None, "") else str(error_message).strip()
        if status == "failed" and not normalized_error_message:
            raise ValueError("error_message is required when status is failed")
        if status != "failed" and normalized_error_message is not None:
            raise ValueError("error_message must be omitted unless status is failed")
        return cls(
            image_id=image_id,
            status=status,
            occurred_at=occurred_at,
            error_message=normalized_error_message,
        )

    def to_body(self) -> bytes:
        return json.dumps(
            {
                "image_id": self.image_id,
                "status": self.status,
                "occurred_at": self.occurred_at,
                "error_message": self.error_message,
            },
            ensure_ascii=False,
        ).encode("utf-8")


@dataclass(frozen=True)
class QueueDelivery:
    delivery_tag: int
    message: ImageJobMessage


class RabbitMqImageJobQueue:
    def __init__(self, amqp_url: str, queue_name: str, max_retries: int = 5, retry_delay: float = 2.0, heartbeat: int = 60) -> None:
        self.parameters = pika.URLParameters(amqp_url)
        self.parameters.heartbeat = heartbeat
        self.queue_name = queue_name
        
        last_exception = None
        for attempt in range(max_retries):
            try:
                self.connection = pika.BlockingConnection(self.parameters)
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue=queue_name, durable=True)
                return
            except pika.exceptions.AMQPConnectionError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue
        
        raise last_exception or ConnectionError("Failed to connect to RabbitMQ")

    def publish(self, message: ImageJobMessage) -> None:
        self.channel.basic_publish(
            exchange="",
            routing_key=self.queue_name,
            body=message.to_body(),
            properties=pika.BasicProperties(delivery_mode=2),
        )

    def publish_result(self, queue_name: str, message: ImageJobResultMessage) -> None:
        self.channel.queue_declare(queue=queue_name, durable=True)
        self.channel.basic_publish(
            exchange="",
            routing_key=queue_name,
            body=message.to_body(),
            properties=pika.BasicProperties(delivery_mode=2),
        )

    def get_batch(self, max_count: int, wait_seconds: float) -> list[QueueDelivery]:
        deadline = time.monotonic() + wait_seconds
        deliveries: list[QueueDelivery] = []
        while len(deliveries) < max_count:
            method, _, body = self.channel.basic_get(queue=self.queue_name, auto_ack=False)
            if method is None:
                if deliveries or time.monotonic() >= deadline:
                    break
                time.sleep(0.1)
                continue
            deliveries.append(
                QueueDelivery(
                    delivery_tag=method.delivery_tag,
                    message=ImageJobMessage.from_body(body),
                )
            )
        return deliveries

    def ack(self, delivery_tag: int) -> None:
        self.channel.basic_ack(delivery_tag=delivery_tag)

    def nack(self, delivery_tag: int, requeue: bool) -> None:
        self.channel.basic_nack(delivery_tag=delivery_tag, requeue=requeue)

    def close(self) -> None:
        if self.connection.is_open:
            self.connection.close()


def _require_non_blank_string(payload: dict[str, object], field_name: str) -> str:
    value = payload.get(field_name)
    if value is None:
        raise ValueError(f"{field_name} is required")
    normalized = str(value).strip()
    if normalized == "":
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _parse_timestamp(value: str) -> None:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("occurred_at must be a valid ISO 8601 timestamp") from exc
