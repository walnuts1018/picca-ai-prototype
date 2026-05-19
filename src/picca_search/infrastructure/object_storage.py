from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any


class SeaweedObjectStorage:
    def __init__(self, s3_client: Any, bucket: str, endpoint_url: str | None = None) -> None:
        self.s3_client = s3_client
        self.bucket = bucket
        self.endpoint_url = endpoint_url

    def uri_for(self, object_key: str) -> str:
        return f"s3://{self.bucket}/{object_key.lstrip('/')}"

    def download_to_tempfile(self, object_key: str) -> Path:
        suffix = Path(object_key).suffix or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
        try:
            self.s3_client.download_file(self.bucket, object_key, str(temporary_path))
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return temporary_path
