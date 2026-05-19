from __future__ import annotations

from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

from picca_search.domain import DenseVector, SparseVector


class DenseModelClient:
    def __init__(self, base_url: str, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=300.0)

    def encode_texts(self, texts: list[str]) -> list[DenseVector]:
        response = self.client.post(f"{self.base_url}/encode/text-batch", json={"texts": texts})
        response.raise_for_status()
        return [DenseVector.create(values) for values in response.json()["vectors"]]

    def encode_images(self, image_paths: list[Path]) -> list[DenseVector]:
        files = [("images", _image_file_tuple(path)) for path in image_paths]
        response = self.client.post(f"{self.base_url}/encode/image-batch", files=files)
        response.raise_for_status()
        return [DenseVector.create(values) for values in response.json()["vectors"]]


class SparseModelClient:
    def __init__(self, base_url: str, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=300.0)

    def encode_texts(self, texts: list[str]) -> list[SparseVector]:
        response = self.client.post(f"{self.base_url}/encode/text-batch", json={"texts": texts})
        response.raise_for_status()
        return [
            SparseVector.create(indices=vector["indices"], values=vector["values"])
            for vector in response.json()["vectors"]
        ]


class OcrModelClient:
    def __init__(self, base_url: str, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=300.0)

    def extract_text(self, image_path: Path) -> str:
        response = self.client.post(
            f"{self.base_url}/extract",
            files={"image": _image_file_tuple(image_path)},
        )
        response.raise_for_status()
        return str(response.json()["text"])


class CaptionModelClient:
    def __init__(self, base_url: str, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=300.0)

    def caption(self, image_path: Path) -> str:
        response = self.client.post(
            f"{self.base_url}/caption",
            files={"image": _image_file_tuple(image_path)},
        )
        response.raise_for_status()
        return str(response.json()["text"])


def _image_file_tuple(path: Path) -> tuple[str, bytes, str]:
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        buffer = BytesIO()
        rgb_image.save(buffer, format="JPEG", quality=90)
    return (path.name or "image.jpg", buffer.getvalue(), "image/jpeg")
