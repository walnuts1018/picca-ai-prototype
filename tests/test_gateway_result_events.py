from __future__ import annotations

import json
from contextlib import contextmanager
import threading

import pytest

from picca_search.domain import DenseVector, SparseVector
from picca_search.gateway import ingestion as ingestion_module
from picca_search.gateway.config import GatewaySettings
from picca_search.gateway.ingestion import GatewayIngestionService, PendingImageJob
from picca_search.gateway import runtime as runtime_module
from picca_search.infrastructure.rabbitmq_queue import ImageJobResultMessage
from picca_search.infrastructure.rabbitmq_queue import ImageJobMessage, QueueDelivery, RabbitMqImageJobQueue


def test_result_message_serializes_failed_event() -> None:
    message = ImageJobResultMessage(
        image_id="debug/a.jpg",
        status="failed",
        occurred_at="2026-05-20T12:00:00Z",
        error_message="caption timeout",
    )

    parsed = ImageJobResultMessage.from_body(message.to_body())

    assert parsed.image_id == "debug/a.jpg"
    assert parsed.status == "failed"
    assert parsed.occurred_at == "2026-05-20T12:00:00Z"
    assert parsed.error_message == "caption timeout"


def test_gateway_settings_reads_result_queue_from_env(monkeypatch) -> None:
    monkeypatch.delenv("RABBITMQ_RESULT_QUEUE", raising=False)

    default_settings = GatewaySettings.from_env()

    assert default_settings.rabbitmq_result_queue == "image_job_results"

    monkeypatch.setenv("RABBITMQ_RESULT_QUEUE", "debug_image_results")

    overridden_settings = GatewaySettings.from_env()

    assert overridden_settings.rabbitmq_result_queue == "debug_image_results"


def test_result_message_rejects_unsupported_status() -> None:
    body = json.dumps(
        {
            "image_id": "debug/a.jpg",
            "status": "completed",
            "occurred_at": "2026-05-20T12:00:00Z",
            "error_message": None,
        }
    ).encode("utf-8")

    with pytest.raises(ValueError, match="status"):
        ImageJobResultMessage.from_body(body)


def test_result_message_rejects_payload_missing_required_fields() -> None:
    body = json.dumps(
        {
            "image_id": "debug/a.jpg",
            "occurred_at": "2026-05-20T12:00:00Z",
        }
    ).encode("utf-8")

    with pytest.raises(ValueError, match="status"):
        ImageJobResultMessage.from_body(body)


def test_result_message_rejects_invalid_timestamp() -> None:
    body = json.dumps(
        {
            "image_id": "debug/a.jpg",
            "status": "indexed",
            "occurred_at": "not-a-timestamp",
            "error_message": None,
        }
    ).encode("utf-8")

    with pytest.raises(ValueError, match="occurred_at"):
        ImageJobResultMessage.from_body(body)


def test_result_message_requires_error_message_for_failed_status() -> None:
    body = json.dumps(
        {
            "image_id": "debug/a.jpg",
            "status": "failed",
            "occurred_at": "2026-05-20T12:00:00Z",
            "error_message": None,
        }
    ).encode("utf-8")

    with pytest.raises(ValueError, match="error_message"):
        ImageJobResultMessage.from_body(body)


def test_result_message_rejects_error_message_for_non_failed_status() -> None:
    body = json.dumps(
        {
            "image_id": "debug/a.jpg",
            "status": "indexed",
            "occurred_at": "2026-05-20T12:00:00Z",
            "error_message": "unexpected",
        }
    ).encode("utf-8")

    with pytest.raises(ValueError, match="error_message"):
        ImageJobResultMessage.from_body(body)


def test_ingestion_outcome_includes_indexed_image_result_event(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "a.jpg"
    image_path.write_bytes(b"image")

    @contextmanager
    def fake_prepare_inference_image(path):
        yield path

    monkeypatch.setattr(ingestion_module, "prepare_inference_image", fake_prepare_inference_image)

    service = GatewayIngestionService(
        storage=_FakeStorage(image_path),
        dense_client=_FakeDenseClient(),
        sparse_client=_FakeSparseClient(),
        ocr_client=_FakeOcrClient("torii"),
        caption_client=_FakeCaptionClient("red gate"),
        index=_FakeIndex(),
    )

    outcome = service.process_jobs([PendingImageJob(delivery_tag=11, image_id="debug/a.jpg")])

    assert outcome.acked_delivery_tags == [11]
    assert outcome.requeue_delivery_tags == []
    assert outcome.dead_letter_delivery_tags == []
    assert len(outcome.image_result_events) == 1
    event = outcome.image_result_events[0]
    assert event.delivery_tag == 11
    assert event.image_id == "debug/a.jpg"
    assert event.status == "indexed"
    assert event.error_message is None


def test_ingestion_outcome_includes_failed_image_result_event_for_dead_letter(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "a.jpg"
    image_path.write_bytes(b"image")

    @contextmanager
    def fake_prepare_inference_image(path):
        yield path

    monkeypatch.setattr(ingestion_module, "prepare_inference_image", fake_prepare_inference_image)

    service = GatewayIngestionService(
        storage=_FakeStorage(image_path),
        dense_client=_FakeDenseClient(),
        sparse_client=_FakeSparseClient(),
        ocr_client=_FakeOcrClient(""),
        caption_client=_FakeCaptionClient(""),
        index=_FakeIndex(),
    )

    outcome = service.process_jobs([PendingImageJob(delivery_tag=12, image_id="debug/b.jpg")])

    assert outcome.acked_delivery_tags == []
    assert outcome.requeue_delivery_tags == []
    assert outcome.dead_letter_delivery_tags == [12]
    assert len(outcome.image_result_events) == 1
    event = outcome.image_result_events[0]
    assert event.delivery_tag == 12
    assert event.image_id == "debug/b.jpg"
    assert event.status == "failed"
    assert event.error_message == "Extracted image text must not be blank"


def test_ingestion_outcome_does_not_include_terminal_event_for_requeue(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "a.jpg"
    image_path.write_bytes(b"image")

    @contextmanager
    def fake_prepare_inference_image(path):
        yield path

    monkeypatch.setattr(ingestion_module, "prepare_inference_image", fake_prepare_inference_image)

    service = GatewayIngestionService(
        storage=_FakeStorage(image_path),
        dense_client=_ExplodingDenseClient(),
        sparse_client=_FakeSparseClient(),
        ocr_client=_FakeOcrClient("torii"),
        caption_client=_FakeCaptionClient("red gate"),
        index=_FakeIndex(),
    )

    outcome = service.process_jobs([PendingImageJob(delivery_tag=13, image_id="debug/c.jpg")])

    assert outcome.acked_delivery_tags == []
    assert outcome.requeue_delivery_tags == [13]
    assert outcome.dead_letter_delivery_tags == []
    assert outcome.image_result_events == []


def test_ingestion_mixed_batch_dead_letters_only_failing_image(tmp_path, monkeypatch) -> None:
    good_path = tmp_path / "good.jpg"
    bad_path = tmp_path / "bad.jpg"
    good_path.write_bytes(b"good")
    bad_path.write_bytes(b"bad")

    @contextmanager
    def fake_prepare_inference_image(path):
        yield path

    monkeypatch.setattr(ingestion_module, "prepare_inference_image", fake_prepare_inference_image)

    service = GatewayIngestionService(
        storage=_MappedStorage({"debug/good.jpg": good_path, "debug/bad.jpg": bad_path}),
        dense_client=_FakeDenseClient(),
        sparse_client=_FakeSparseClient(),
        ocr_client=_MappedOcrClient({str(good_path): "torii", str(bad_path): ""}),
        caption_client=_MappedCaptionClient({str(good_path): "red gate", str(bad_path): ""}),
        index=_RecordingIndex(),
    )

    outcome = service.process_jobs(
        [
            PendingImageJob(delivery_tag=21, image_id="debug/good.jpg"),
            PendingImageJob(delivery_tag=22, image_id="debug/bad.jpg"),
        ]
    )

    assert outcome.acked_delivery_tags == [21]
    assert outcome.requeue_delivery_tags == []
    assert outcome.dead_letter_delivery_tags == [22]
    assert [(event.delivery_tag, event.image_id, event.status) for event in outcome.image_result_events] == [
        (22, "debug/bad.jpg", "failed"),
        (21, "debug/good.jpg", "indexed"),
    ]
    assert outcome.image_result_events[0].error_message == "Extracted image text must not be blank"


def test_ingestion_batch_level_failure_requeues_only_prepared_subset(tmp_path, monkeypatch) -> None:
    good_path = tmp_path / "good.jpg"
    bad_path = tmp_path / "bad.jpg"
    good_path.write_bytes(b"good")
    bad_path.write_bytes(b"bad")

    @contextmanager
    def fake_prepare_inference_image(path):
        yield path

    monkeypatch.setattr(ingestion_module, "prepare_inference_image", fake_prepare_inference_image)

    service = GatewayIngestionService(
        storage=_MappedStorage({"debug/good.jpg": good_path, "debug/bad.jpg": bad_path}),
        dense_client=_ExplodingDenseClient(),
        sparse_client=_FakeSparseClient(),
        ocr_client=_MappedOcrClient({str(good_path): "torii", str(bad_path): ""}),
        caption_client=_MappedCaptionClient({str(good_path): "red gate", str(bad_path): ""}),
        index=_RecordingIndex(),
    )

    outcome = service.process_jobs(
        [
            PendingImageJob(delivery_tag=31, image_id="debug/good.jpg"),
            PendingImageJob(delivery_tag=32, image_id="debug/bad.jpg"),
        ]
    )

    assert outcome.acked_delivery_tags == []
    assert outcome.requeue_delivery_tags == [31]
    assert outcome.dead_letter_delivery_tags == [32]
    assert [(event.delivery_tag, event.image_id, event.status) for event in outcome.image_result_events] == [
        (32, "debug/bad.jpg", "failed"),
    ]
    assert outcome.image_result_events[0].error_message == "Extracted image text must not be blank"


def test_ingestion_mixed_batch_requeues_only_image_with_unexpected_preparation_failure(tmp_path, monkeypatch) -> None:
    good_path = tmp_path / "good.jpg"
    boom_path = tmp_path / "boom.jpg"
    good_path.write_bytes(b"good")
    boom_path.write_bytes(b"boom")

    @contextmanager
    def fake_prepare_inference_image(path):
        yield path

    monkeypatch.setattr(ingestion_module, "prepare_inference_image", fake_prepare_inference_image)

    service = GatewayIngestionService(
        storage=_MappedStorage({"debug/good.jpg": good_path, "debug/boom.jpg": boom_path}),
        dense_client=_FakeDenseClient(),
        sparse_client=_FakeSparseClient(),
        ocr_client=_MappedOcrClient({str(good_path): "torii", str(boom_path): "ignored"}),
        caption_client=_ExplodingCaptionClient({str(good_path): "red gate", str(boom_path): RuntimeError("caption timeout")}),
        index=_RecordingIndex(),
    )

    outcome = service.process_jobs(
        [
            PendingImageJob(delivery_tag=41, image_id="debug/boom.jpg"),
            PendingImageJob(delivery_tag=42, image_id="debug/good.jpg"),
        ]
    )

    assert outcome.acked_delivery_tags == [42]
    assert outcome.requeue_delivery_tags == [41]
    assert outcome.dead_letter_delivery_tags == []
    assert [(event.delivery_tag, event.image_id, event.status) for event in outcome.image_result_events] == [
        (42, "debug/good.jpg", "indexed"),
    ]
    assert [document.image_id.value for document in service.index.documents] == ["debug/good.jpg"]


def test_rabbitmq_queue_can_publish_result_message_to_named_queue() -> None:
    published = []

    class _FakeChannel:
        def queue_declare(self, **kwargs) -> None:
            return None

        def basic_publish(self, **kwargs) -> None:
            published.append(kwargs)

    queue = object.__new__(RabbitMqImageJobQueue)
    queue.channel = _FakeChannel()

    message = ImageJobResultMessage(
        image_id="debug/a.jpg",
        status="indexed",
        occurred_at="2026-05-20T12:00:00Z",
    )

    queue.publish_result("debug_image_results", message)

    assert len(published) == 1
    assert published[0]["exchange"] == ""
    assert published[0]["routing_key"] == "debug_image_results"
    assert published[0]["body"] == message.to_body()
    assert published[0]["properties"].delivery_mode == 2


def test_runtime_publishes_terminal_indexed_event_before_ack(monkeypatch) -> None:
    stop_event = threading.Event()
    published_messages = []
    acked = []
    closed = []
    call_order = []

    class _FakeQueue:
        def __init__(self, *_args, **_kwargs) -> None:
            self._returned_batch = False

        def get_batch(self, _max_count: int, _wait_seconds: float) -> list[QueueDelivery]:
            if self._returned_batch:
                stop_event.set()
                return []
            self._returned_batch = True
            return [QueueDelivery(delivery_tag=21, message=ImageJobMessage(image_id="debug/a.jpg"))]

        def publish_result(self, queue_name: str, message: ImageJobResultMessage) -> None:
            call_order.append(("publish", message.status, message.image_id))
            published_messages.append((queue_name, message))

        def ack(self, delivery_tag: int) -> None:
            call_order.append(("ack", delivery_tag))
            acked.append(delivery_tag)

        def nack(self, delivery_tag: int, requeue: bool) -> None:
            raise AssertionError(f"unexpected nack: {delivery_tag=} {requeue=}")

        def close(self) -> None:
            closed.append(True)

    class _FakeIngestion:
        def __init__(self, **_kwargs) -> None:
            pass

        def process_jobs(self, jobs):
            stop_event.set()
            assert [job.image_id for job in jobs] == ["debug/a.jpg"]
            return ingestion_module.IngestionOutcome(
                acked_delivery_tags=[21],
                requeue_delivery_tags=[],
                dead_letter_delivery_tags=[],
                image_result_events=[
                    ingestion_module.ImageResultEvent(
                        delivery_tag=21,
                        image_id="debug/a.jpg",
                        status="indexed",
                    )
                ],
            )

    monkeypatch.setattr(runtime_module, "_create_s3_client", lambda settings: object())
    monkeypatch.setattr(runtime_module, "SeaweedObjectStorage", lambda **kwargs: object())
    monkeypatch.setattr(runtime_module, "DenseModelClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "SparseModelClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "OcrModelClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "CaptionModelClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "QdrantClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "QdrantImageIndex", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "RabbitMqImageJobQueue", _FakeQueue)
    monkeypatch.setattr(runtime_module, "GatewayIngestionService", _FakeIngestion)

    settings = GatewaySettings(rabbitmq_result_queue="debug_image_results")

    runtime_module._run_consumer_loop(settings, stop_event)

    assert acked == [21]
    assert closed == [True]
    assert [queue_name for queue_name, _message in published_messages] == [
        "debug_image_results",
        "debug_image_results",
    ]
    assert [message.status for _queue_name, message in published_messages] == ["processing", "indexed"]
    assert [message.image_id for _queue_name, message in published_messages] == ["debug/a.jpg", "debug/a.jpg"]
    assert call_order == [
        ("publish", "processing", "debug/a.jpg"),
        ("publish", "indexed", "debug/a.jpg"),
        ("ack", 21),
    ]


def test_runtime_does_not_emit_second_processing_before_requeue_nack(monkeypatch) -> None:
    stop_event = threading.Event()
    call_order = []
    closed = []

    class _FakeQueue:
        def __init__(self, *_args, **_kwargs) -> None:
            self._returned_batch = False

        def get_batch(self, _max_count: int, _wait_seconds: float) -> list[QueueDelivery]:
            if self._returned_batch:
                stop_event.set()
                return []
            self._returned_batch = True
            return [QueueDelivery(delivery_tag=31, message=ImageJobMessage(image_id="debug/retry.jpg"))]

        def publish_result(self, queue_name: str, message: ImageJobResultMessage) -> None:
            call_order.append(("publish", queue_name, message.status, message.image_id, message.error_message))

        def nack(self, delivery_tag: int, requeue: bool) -> None:
            call_order.append(("nack", delivery_tag, requeue))

        def close(self) -> None:
            closed.append(True)

    class _FakeIngestion:
        def __init__(self, **_kwargs) -> None:
            pass

        def process_jobs(self, jobs):
            stop_event.set()
            return ingestion_module.IngestionOutcome(
                acked_delivery_tags=[],
                requeue_delivery_tags=[31],
                dead_letter_delivery_tags=[],
                image_result_events=[],
            )

    monkeypatch.setattr(runtime_module, "_create_s3_client", lambda settings: object())
    monkeypatch.setattr(runtime_module, "SeaweedObjectStorage", lambda **kwargs: object())
    monkeypatch.setattr(runtime_module, "DenseModelClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "SparseModelClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "OcrModelClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "CaptionModelClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "QdrantClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "QdrantImageIndex", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "RabbitMqImageJobQueue", _FakeQueue)
    monkeypatch.setattr(runtime_module, "GatewayIngestionService", _FakeIngestion)

    settings = GatewaySettings(rabbitmq_result_queue="debug_image_results")

    runtime_module._run_consumer_loop(settings, stop_event)

    assert closed == [True]
    assert call_order == [
        ("publish", "debug_image_results", "processing", "debug/retry.jpg", None),
        ("nack", 31, True),
    ]


def test_runtime_synthesizes_failed_event_before_dead_letter_nack(monkeypatch) -> None:
    stop_event = threading.Event()
    call_order = []
    closed = []

    class _FakeQueue:
        def __init__(self, *_args, **_kwargs) -> None:
            self._returned_batch = False

        def get_batch(self, _max_count: int, _wait_seconds: float) -> list[QueueDelivery]:
            if self._returned_batch:
                stop_event.set()
                return []
            self._returned_batch = True
            return [QueueDelivery(delivery_tag=41, message=ImageJobMessage(image_id="debug/dead.jpg"))]

        def publish_result(self, queue_name: str, message: ImageJobResultMessage) -> None:
            call_order.append(("publish", queue_name, message.status, message.image_id, message.error_message))

        def ack(self, delivery_tag: int) -> None:
            raise AssertionError(f"unexpected ack: {delivery_tag=}")

        def nack(self, delivery_tag: int, requeue: bool) -> None:
            call_order.append(("nack", delivery_tag, requeue))

        def close(self) -> None:
            closed.append(True)

    class _FakeIngestion:
        def __init__(self, **_kwargs) -> None:
            pass

        def process_jobs(self, jobs):
            stop_event.set()
            return ingestion_module.IngestionOutcome(
                acked_delivery_tags=[],
                requeue_delivery_tags=[],
                dead_letter_delivery_tags=[41],
                image_result_events=[],
            )

    monkeypatch.setattr(runtime_module, "_create_s3_client", lambda settings: object())
    monkeypatch.setattr(runtime_module, "SeaweedObjectStorage", lambda **kwargs: object())
    monkeypatch.setattr(runtime_module, "DenseModelClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "SparseModelClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "OcrModelClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "CaptionModelClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "QdrantClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "QdrantImageIndex", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "RabbitMqImageJobQueue", _FakeQueue)
    monkeypatch.setattr(runtime_module, "GatewayIngestionService", _FakeIngestion)

    settings = GatewaySettings(rabbitmq_result_queue="debug_image_results")

    runtime_module._run_consumer_loop(settings, stop_event)

    assert closed == [True]
    assert call_order == [
        ("publish", "debug_image_results", "processing", "debug/dead.jpg", None),
        (
            "publish",
            "debug_image_results",
            "failed",
            "debug/dead.jpg",
            "dead-lettered without explicit ingestion failure event",
        ),
        ("nack", 41, False),
    ]


def test_runtime_publish_failure_prevents_terminal_disposition(monkeypatch) -> None:
    stop_event = threading.Event()
    closed = []
    acked = []
    nacks = []
    published_statuses = []

    class _ExitCalled(Exception):
        pass

    class _FakeQueue:
        def __init__(self, *_args, **_kwargs) -> None:
            self._returned_batch = False

        def get_batch(self, _max_count: int, _wait_seconds: float) -> list[QueueDelivery]:
            if self._returned_batch:
                stop_event.set()
                return []
            self._returned_batch = True
            return [QueueDelivery(delivery_tag=51, message=ImageJobMessage(image_id="debug/fail.jpg"))]

        def publish_result(self, queue_name: str, message: ImageJobResultMessage) -> None:
            published_statuses.append(message.status)
            if message.status == "indexed":
                raise RuntimeError("publish failed")

        def ack(self, delivery_tag: int) -> None:
            acked.append(delivery_tag)

        def nack(self, delivery_tag: int, requeue: bool) -> None:
            nacks.append((delivery_tag, requeue))

        def close(self) -> None:
            closed.append(True)

    class _FakeIngestion:
        def __init__(self, **_kwargs) -> None:
            pass

        def process_jobs(self, jobs):
            stop_event.set()
            return ingestion_module.IngestionOutcome(
                acked_delivery_tags=[51],
                requeue_delivery_tags=[],
                dead_letter_delivery_tags=[],
                image_result_events=[
                    ingestion_module.ImageResultEvent(
                        delivery_tag=51,
                        image_id="debug/fail.jpg",
                        status="indexed",
                    )
                ],
            )

    monkeypatch.setattr(runtime_module, "_create_s3_client", lambda settings: object())
    monkeypatch.setattr(runtime_module, "SeaweedObjectStorage", lambda **kwargs: object())
    monkeypatch.setattr(runtime_module, "DenseModelClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "SparseModelClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "OcrModelClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "CaptionModelClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "QdrantClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "QdrantImageIndex", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "RabbitMqImageJobQueue", _FakeQueue)
    monkeypatch.setattr(runtime_module, "GatewayIngestionService", _FakeIngestion)
    monkeypatch.setattr(runtime_module.os, "_exit", lambda code: (_ for _ in ()).throw(_ExitCalled(code)))

    settings = GatewaySettings(rabbitmq_result_queue="debug_image_results")

    with pytest.raises(_ExitCalled):
        runtime_module._run_consumer_loop(settings, stop_event)

    assert closed == [True]
    assert published_statuses == ["processing", "indexed"]
    assert acked == []
    assert nacks == []
    
    
def test_runtime_publish_failure_prevents_dead_letter_disposition(monkeypatch) -> None:
    stop_event = threading.Event()
    closed = []
    nacks = []
    published_statuses = []

    class _ExitCalled(Exception):
        pass

    class _FakeQueue:
        def __init__(self, *_args, **_kwargs) -> None:
            self._returned_batch = False

        def get_batch(self, _max_count: int, _wait_seconds: float) -> list[QueueDelivery]:
            if self._returned_batch:
                stop_event.set()
                return []
            self._returned_batch = True
            return [QueueDelivery(delivery_tag=61, message=ImageJobMessage(image_id="debug/dead-fail.jpg"))]

        def publish_result(self, queue_name: str, message: ImageJobResultMessage) -> None:
            published_statuses.append(message.status)
            if message.status == "failed":
                raise RuntimeError("publish failed")

        def ack(self, delivery_tag: int) -> None:
            raise AssertionError(f"unexpected ack: {delivery_tag=}")

        def nack(self, delivery_tag: int, requeue: bool) -> None:
            nacks.append((delivery_tag, requeue))

        def close(self) -> None:
            closed.append(True)

    class _FakeIngestion:
        def __init__(self, **_kwargs) -> None:
            pass

        def process_jobs(self, jobs):
            stop_event.set()
            return ingestion_module.IngestionOutcome(
                acked_delivery_tags=[],
                requeue_delivery_tags=[],
                dead_letter_delivery_tags=[61],
                image_result_events=[],
            )

    monkeypatch.setattr(runtime_module, "_create_s3_client", lambda settings: object())
    monkeypatch.setattr(runtime_module, "SeaweedObjectStorage", lambda **kwargs: object())
    monkeypatch.setattr(runtime_module, "DenseModelClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "SparseModelClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "OcrModelClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "CaptionModelClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "QdrantClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "QdrantImageIndex", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime_module, "RabbitMqImageJobQueue", _FakeQueue)
    monkeypatch.setattr(runtime_module, "GatewayIngestionService", _FakeIngestion)
    monkeypatch.setattr(runtime_module.os, "_exit", lambda code: (_ for _ in ()).throw(_ExitCalled(code)))

    settings = GatewaySettings(rabbitmq_result_queue="debug_image_results")

    with pytest.raises(_ExitCalled):
        runtime_module._run_consumer_loop(settings, stop_event)

    assert closed == [True]
    assert published_statuses == ["processing", "failed"]
    assert nacks == []


class _FakeStorage:
    def __init__(self, image_path) -> None:
        self.image_path = image_path

    def download_to_tempfile(self, image_id: str):
        return self.image_path

    def uri_for(self, image_id: str) -> str:
        return f"s3://images/{image_id}"


class _MappedStorage:
    def __init__(self, image_paths: dict[str, object]) -> None:
        self.image_paths = image_paths

    def download_to_tempfile(self, image_id: str):
        return self.image_paths[image_id]

    def uri_for(self, image_id: str) -> str:
        return f"s3://images/{image_id}"


class _FakeDenseClient:
    def encode_images(self, paths) -> list[DenseVector]:
        return [DenseVector.create([1.0]) for _ in paths]


class _ExplodingDenseClient:
    def encode_images(self, paths) -> list[DenseVector]:
        raise RuntimeError("dense timeout")


class _FakeSparseClient:
    def encode_texts(self, texts) -> list[SparseVector]:
        return [SparseVector.create([1], [1.0]) for _ in texts]


class _FakeOcrClient:
    def __init__(self, text: str) -> None:
        self.text = text

    def extract_text(self, path) -> str:
        return self.text


class _MappedOcrClient:
    def __init__(self, texts: dict[str, str]) -> None:
        self.texts = texts

    def extract_text(self, path) -> str:
        return self.texts[str(path)]


class _FakeCaptionClient:
    def __init__(self, caption: str) -> None:
        self.caption_text = caption

    def caption(self, path) -> str:
        return self.caption_text


class _MappedCaptionClient:
    def __init__(self, captions: dict[str, str]) -> None:
        self.captions = captions

    def caption(self, path) -> str:
        return self.captions[str(path)]


class _ExplodingCaptionClient:
    def __init__(self, captions: dict[str, str | Exception]) -> None:
        self.captions = captions

    def caption(self, path) -> str:
        result = self.captions[str(path)]
        if isinstance(result, Exception):
            raise result
        return result


class _FakeIndex:
    def upsert(self, documents) -> None:
        return None


class _RecordingIndex:
    def __init__(self) -> None:
        self.documents = []

    def upsert(self, documents) -> None:
        self.documents.extend(documents)
