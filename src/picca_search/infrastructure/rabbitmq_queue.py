from __future__ import annotations

import json
import time
from dataclasses import dataclass

import pika


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
class QueueDelivery:
    delivery_tag: int
    message: ImageJobMessage


class RabbitMqImageJobQueue:
    def __init__(self, amqp_url: str, queue_name: str) -> None:
        self.parameters = pika.URLParameters(amqp_url)
        self.connection = pika.BlockingConnection(self.parameters)
        self.channel = self.connection.channel()
        self.queue_name = queue_name
        self.channel.queue_declare(queue=queue_name, durable=True)

    def publish(self, message: ImageJobMessage) -> None:
        self.channel.basic_publish(
            exchange="",
            routing_key=self.queue_name,
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
