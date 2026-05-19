from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1)
    dense_weight: float | None = Field(default=None, gt=0)
    ocr_weight: float | None = Field(default=None, gt=0)
    florence_weight: float | None = Field(default=None, gt=0)
    include_diagnostics: bool = False

    def weights_tuple(self, default_weights: tuple[float, float, float]) -> tuple[float, float, float]:
        dense, ocr, florence = default_weights
        return (
            self.dense_weight if self.dense_weight is not None else dense,
            self.ocr_weight if self.ocr_weight is not None else ocr,
            self.florence_weight if self.florence_weight is not None else florence,
        )


class SearchResultPayload(BaseModel):
    image_id: str
    score: float
    path: str
    text: str
    ocr_text: str | None = None
    caption: str | None = None


class SearchResponse(BaseModel):
    query: str
    limit: int
    weights: dict[str, float]
    results: list[SearchResultPayload]
    diagnostics: dict[str, list[SearchResultPayload]] | None = None
