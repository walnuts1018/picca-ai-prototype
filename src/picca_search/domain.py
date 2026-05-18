from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


@dataclass(frozen=True)
class ImageId:
    value: str

    @classmethod
    def from_path(cls, path: Path) -> "ImageId":
        resolved = str(path.expanduser().resolve())
        digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()
        return cls(str(uuid.uuid5(uuid.NAMESPACE_URL, digest)))


@dataclass(frozen=True)
class ImagePath:
    value: Path

    @classmethod
    def create(cls, path: Path | str) -> "ImagePath":
        value = Path(path).expanduser()
        if not value.exists():
            raise ValueError(f"Image file does not exist: {value}")
        if not value.is_file():
            raise ValueError(f"Image path is not a file: {value}")
        if value.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {value.suffix}")
        return cls(value)


@dataclass(frozen=True)
class DenseVector:
    values: tuple[float, ...]

    @classmethod
    def create(cls, values: list[float] | tuple[float, ...]) -> "DenseVector":
        if len(values) == 0:
            raise ValueError("Dense vector must not be empty")
        return cls(tuple(float(value) for value in values))

    @property
    def dimension(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class SparseVector:
    indices: tuple[int, ...]
    values: tuple[float, ...]

    @classmethod
    def create(
        cls,
        indices: list[int] | tuple[int, ...],
        values: list[float] | tuple[float, ...],
    ) -> "SparseVector":
        if len(indices) == 0 or len(values) == 0:
            raise ValueError("Sparse vector must not be empty")
        if len(indices) != len(values):
            raise ValueError("Sparse vector indices and values must have the same length")
        pairs = sorted((int(index), float(value)) for index, value in zip(indices, values))
        if any(index < 0 for index, _ in pairs):
            raise ValueError("Sparse vector indices must be non-negative")
        if len({index for index, _ in pairs}) != len(pairs):
            raise ValueError("Sparse vector indices must be unique")
        return cls(
            indices=tuple(index for index, _ in pairs),
            values=tuple(value for _, value in pairs),
        )


@dataclass(frozen=True)
class SearchQuery:
    text: str

    @classmethod
    def create(cls, text: str) -> "SearchQuery":
        normalized = text.strip()
        if normalized == "":
            raise ValueError("Search query must not be blank")
        return cls(normalized)


@dataclass(frozen=True)
class ExtractedImageText:
    combined: str

    @classmethod
    def create(cls, ocr_text: str, caption: str) -> "ExtractedImageText":
        parts = [
            normalized
            for text in (ocr_text, caption)
            if (normalized := _normalize_text_block(text)) != ""
        ]
        if len(parts) == 0:
            raise ValueError("Extracted image text must not be blank")
        return cls("\n".join(parts))


@dataclass(frozen=True)
class ImageDocument:
    image_id: ImageId
    image_path: ImagePath
    dense_vector: DenseVector
    sparse_vector: SparseVector
    text: str

    @classmethod
    def create(
        cls,
        image_id: ImageId,
        image_path: ImagePath,
        dense_vector: DenseVector,
        sparse_vector: SparseVector,
        text: str,
    ) -> "ImageDocument":
        return cls(
            image_id=image_id,
            image_path=image_path,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            text=text.strip(),
        )

    @property
    def payload(self) -> dict[str, str]:
        return {
            "path": str(self.image_path.value),
            "text": self.text,
        }


@dataclass(frozen=True)
class SearchResult:
    image_id: ImageId
    score: float
    payload: dict[str, object]


def _normalize_text_block(text: str) -> str:
    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.splitlines()
        if line.strip() != ""
    ]
    return "\n".join(lines)
