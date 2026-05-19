from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field


class TextBatchRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)


class DenseVectorsResponse(BaseModel):
    vectors: list[list[float]]


class SparseVectorItem(BaseModel):
    indices: list[int]
    values: list[float]


class SparseVectorsResponse(BaseModel):
    vectors: list[SparseVectorItem]


class TextResponse(BaseModel):
    text: str


def create_service_app(title: str) -> FastAPI:
    return FastAPI(title=title)


async def upload_to_tempfile(upload: UploadFile = File(...)) -> Path:
    suffix = Path(upload.filename or "upload.bin").suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
        temporary_file.write(await upload.read())
    if temporary_path.stat().st_size == 0:
        temporary_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return temporary_path
