from __future__ import annotations

import json

import pytest

from picca_search.gateway.config import GatewaySettings
from picca_search.infrastructure.rabbitmq_queue import ImageJobResultMessage


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
