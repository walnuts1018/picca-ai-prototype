from __future__ import annotations

from picca_search.gateway.config import GatewaySettings
from picca_search.gateway.runtime import _create_s3_client


def test_gateway_settings_uses_static_s3_credentials_by_default(monkeypatch) -> None:
    for name in (
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "S3_ENDPOINT_URL",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ENDPOINT_URL_STS",
        "AWS_ENDPOINT_URL_S3",
        "AWS_REGION",
        "AWS_ROLE_ARN",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = GatewaySettings.from_env()

    assert settings.s3_access_key_id == "seaweedfs"
    assert settings.s3_secret_access_key == "seaweedfs"
    assert settings.uses_web_identity is False
    assert settings.resolved_s3_endpoint_url == "http://seaweedfs-s3:8333"


def test_gateway_settings_disables_default_static_credentials_for_web_identity(monkeypatch) -> None:
    monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", "/var/run/secrets/sts.seaweedfs.com/serviceaccount/token")
    monkeypatch.setenv("AWS_ENDPOINT_URL_STS", "https://seaweedfs.local.walnuts.dev")
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "https://seaweedfs.local.walnuts.dev")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ROLE_ARN", "arn:aws:iam::role/ipu")
    monkeypatch.delenv("S3_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("S3_SECRET_ACCESS_KEY", raising=False)

    settings = GatewaySettings.from_env()

    assert settings.aws_web_identity_token_file == "/var/run/secrets/sts.seaweedfs.com/serviceaccount/token"
    assert settings.aws_endpoint_url_sts == "https://seaweedfs.local.walnuts.dev"
    assert settings.aws_endpoint_url_s3 == "https://seaweedfs.local.walnuts.dev"
    assert settings.aws_region == "us-east-1"
    assert settings.aws_role_arn == "arn:aws:iam::role/ipu"
    assert settings.uses_web_identity is True
    assert settings.s3_access_key_id is None
    assert settings.s3_secret_access_key is None
    assert settings.resolved_s3_endpoint_url == "https://seaweedfs.local.walnuts.dev"


def test_create_s3_client_omits_static_credentials_for_web_identity(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSession:
        def __init__(self, region_name=None) -> None:
            captured["session_region_name"] = region_name

        def client(self, service_name: str, **kwargs):
            captured["service_name"] = service_name
            captured["client_kwargs"] = kwargs
            return object()

    monkeypatch.setattr("picca_search.gateway.runtime.boto3.session.Session", FakeSession)
    settings = GatewaySettings(
        s3_endpoint_url="http://seaweedfs-s3:8333",
        s3_access_key_id=None,
        s3_secret_access_key=None,
        aws_web_identity_token_file="/var/run/secrets/sts.seaweedfs.com/serviceaccount/token",
        aws_endpoint_url_sts="https://seaweedfs.local.walnuts.dev",
        aws_endpoint_url_s3="https://seaweedfs.local.walnuts.dev",
        aws_region="us-east-1",
        aws_role_arn="arn:aws:iam::role/ipu",
    )

    _create_s3_client(settings)

    assert captured["session_region_name"] == "us-east-1"
    assert captured["service_name"] == "s3"
    assert captured["client_kwargs"] == {
        "endpoint_url": "https://seaweedfs.local.walnuts.dev",
        "region_name": "us-east-1",
    }


def test_create_s3_client_keeps_static_credentials_when_configured(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSession:
        def __init__(self, region_name=None) -> None:
            captured["session_region_name"] = region_name

        def client(self, service_name: str, **kwargs):
            captured["service_name"] = service_name
            captured["client_kwargs"] = kwargs
            return object()

    monkeypatch.setattr("picca_search.gateway.runtime.boto3.session.Session", FakeSession)
    settings = GatewaySettings(
        s3_endpoint_url="http://seaweedfs-s3:8333",
        s3_access_key_id="seaweedfs",
        s3_secret_access_key="seaweedfs",
        aws_region="us-east-1",
    )

    _create_s3_client(settings)

    assert captured["session_region_name"] == "us-east-1"
    assert captured["service_name"] == "s3"
    assert captured["client_kwargs"] == {
        "endpoint_url": "http://seaweedfs-s3:8333",
        "region_name": "us-east-1",
        "aws_access_key_id": "seaweedfs",
        "aws_secret_access_key": "seaweedfs",
    }
