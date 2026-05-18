from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from picca_search.domain import DenseVector, ImageDocument, ImageId, ImagePath, SparseVector
from scripts import ingest_images, search_images


class FakeDenseEncoder:
    def encode_image(self, image_path: Path) -> DenseVector:
        return DenseVector.create([0.1, 0.2])

    def encode_text(self, text: str) -> DenseVector:
        return DenseVector.create([0.1, 0.2])


class FakeSparseEncoder:
    def encode_text(self, text: str) -> SparseVector:
        return SparseVector.create([1, 2], [0.3, 0.4])


class FakeOcr:
    def extract_text(self, image_path: Path) -> str:
        return f"ocr:{image_path.name}"


class FakeCaptioner:
    def caption(self, image_path: Path) -> str:
        return f"caption:{image_path.name}"


class RecordingIndex:
    def __init__(self) -> None:
        self.upsert_calls: list[list[ImageDocument]] = []
        self.search_calls: list[tuple[DenseVector, SparseVector, int]] = []
        self.search_with_diagnostics_calls: list[tuple[DenseVector, SparseVector, int]] = []

    def upsert(self, documents: list[ImageDocument]) -> None:
        self.upsert_calls.append(documents)

    def search(self, query_dense: DenseVector, query_sparse: SparseVector, limit: int):
        self.search_calls.append((query_dense, query_sparse, limit))
        return []

    def search_with_diagnostics(
        self, query_dense: DenseVector, query_sparse: SparseVector, limit: int
    ):
        self.search_with_diagnostics_calls.append((query_dense, query_sparse, limit))
        return type("Diagnostics", (), {"dense": [], "sparse": [], "fused": []})()


def make_image(path: Path) -> Path:
    path.write_bytes(b"fake")
    return path


def test_ingest_images_flushes_batches(tmp_path: Path, monkeypatch) -> None:
    images = [
        make_image(tmp_path / "a.jpg"),
        make_image(tmp_path / "b.jpg"),
        make_image(tmp_path / "c.jpg"),
    ]
    index = RecordingIndex()

    @contextmanager
    def passthrough_prepare_inference_image(image_path: Path):
        yield image_path

    monkeypatch.setattr(
        ingest_images,
        "prepare_inference_image",
        passthrough_prepare_inference_image,
    )

    ingest_images.ingest_images(
        image_paths=images,
        ocr_text_extractor=FakeOcr(),
        image_captioner=FakeCaptioner(),
        image_dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder(),
        image_index=index,
        batch_size=2,
    )

    assert [len(batch) for batch in index.upsert_calls] == [2, 1]


def test_search_main_uses_fast_path_without_diagnostics(monkeypatch, capsys) -> None:
    index = RecordingIndex()
    monkeypatch.setattr(search_images, "WaonSiglipEncoder", lambda device=None: FakeDenseEncoder())
    monkeypatch.setattr(
        search_images, "SpladeJapaneseSparseEncoder", lambda device=None: FakeSparseEncoder()
    )
    monkeypatch.setattr(search_images, "QdrantImageIndex", lambda client, collection: index)
    monkeypatch.setattr(search_images, "QdrantClient", lambda url: object())
    monkeypatch.setattr(
        "sys.argv",
        ["search_images.py", "hello", "--collection", "images"],
    )

    search_images.main()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(index.search_calls) == 1
    assert len(index.search_with_diagnostics_calls) == 0
