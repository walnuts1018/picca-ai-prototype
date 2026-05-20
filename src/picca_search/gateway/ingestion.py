from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile

from picca_search.domain import ExtractedImageText, ImageDocument, ImageId, ImagePath, ImageSourcePath
from picca_search.infrastructure.image_preprocessing import prepare_inference_image
from picca_search.infrastructure.model_client import (
    CaptionModelClient,
    DenseModelClient,
    OcrModelClient,
    SparseModelClient,
)
from picca_search.infrastructure.object_storage import SeaweedObjectStorage
from picca_search.infrastructure.qdrant_index import QdrantImageIndex


@dataclass(frozen=True)
class ImageResultEvent:
    delivery_tag: int
    image_id: str
    status: str
    error_message: str | None = None


@dataclass(frozen=True)
class IngestionOutcome:
    acked_delivery_tags: list[int]
    requeue_delivery_tags: list[int]
    dead_letter_delivery_tags: list[int]
    image_result_events: list[ImageResultEvent]


@dataclass(frozen=True)
class PendingImageJob:
    delivery_tag: int
    image_id: str


@dataclass(frozen=True)
class _PreparedImageJob:
    job: PendingImageJob
    local_path: Path
    dense_path: Path
    extracted: ExtractedImageText


class GatewayIngestionService:
    def __init__(
        self,
        *,
        storage: SeaweedObjectStorage,
        dense_client: DenseModelClient,
        sparse_client: SparseModelClient,
        ocr_client: OcrModelClient,
        caption_client: CaptionModelClient,
        index: QdrantImageIndex,
    ) -> None:
        self.storage = storage
        self.dense_client = dense_client
        self.sparse_client = sparse_client
        self.ocr_client = ocr_client
        self.caption_client = caption_client
        self.index = index

    def process_jobs(self, jobs: list[PendingImageJob]) -> IngestionOutcome:
        acked: list[int] = []
        requeue: list[int] = []
        dead_letter: list[int] = []
        image_result_events: list[ImageResultEvent] = []
        documents: list[ImageDocument] = []
        prepared_jobs: list[_PreparedImageJob] = []
        florence_texts: list[str] = []
        ocr_texts_for_sparse: list[str] = []
        temp_paths_to_cleanup: list[Path] = []

        for job in jobs:
            try:
                local_path = self.storage.download_to_tempfile(job.image_id)
                temp_paths_to_cleanup.append(local_path)
                with prepare_inference_image(local_path) as inference_path:
                    ocr_text = self.ocr_client.extract_text(inference_path)
                    caption = self.caption_client.caption(inference_path)
                    extracted = ExtractedImageText.create(ocr_text=ocr_text, caption=caption)
                    if inference_path == local_path:
                        dense_path = local_path
                    else:
                        with tempfile.NamedTemporaryFile(
                            suffix=inference_path.suffix,
                            delete=False,
                        ) as temporary_file:
                            retained_path = Path(temporary_file.name)
                        shutil.copy2(inference_path, retained_path)
                        temp_paths_to_cleanup.append(retained_path)
                        dense_path = retained_path
                    prepared_jobs.append(
                        _PreparedImageJob(
                            job=job,
                            local_path=local_path,
                            dense_path=dense_path,
                            extracted=extracted,
                        )
                    )
                    florence_texts.append(extracted.caption or extracted.combined)
                    if extracted.ocr_text != "":
                        ocr_texts_for_sparse.append(extracted.ocr_text)
            except (OSError, ValueError) as exc:
                dead_letter.append(job.delivery_tag)
                image_result_events.append(
                    ImageResultEvent(
                        delivery_tag=job.delivery_tag,
                        image_id=job.image_id,
                        status="failed",
                        error_message=str(exc),
                    )
                )
            except Exception:
                requeue.append(job.delivery_tag)

        try:
            if not prepared_jobs:
                return IngestionOutcome(acked, requeue, dead_letter, image_result_events)

            dense_vectors = self.dense_client.encode_images([item.dense_path for item in prepared_jobs])
            florence_sparse_vectors = self.sparse_client.encode_texts(florence_texts)
            ocr_sparse_vectors = self.sparse_client.encode_texts(ocr_texts_for_sparse)
            next_ocr_sparse = iter(ocr_sparse_vectors)
            for prepared, dense_vector, florence_sparse in zip(
                prepared_jobs,
                dense_vectors,
                florence_sparse_vectors,
                strict=True,
            ):
                documents.append(
                    ImageDocument.create(
                        image_id=ImageId.from_object_key(prepared.job.image_id),
                        image_path=ImagePath.create(prepared.local_path),
                        source_path=ImageSourcePath.create(self.storage.uri_for(prepared.job.image_id)),
                        dense_vector=dense_vector,
                        florence_sparse_vector=florence_sparse,
                        text=prepared.extracted.combined,
                        ocr_sparse_vector=next(next_ocr_sparse) if prepared.extracted.ocr_text != "" else None,
                        ocr_text=prepared.extracted.ocr_text,
                        caption=prepared.extracted.caption,
                    )
                )

            self.index.upsert(documents)
            acked.extend(item.job.delivery_tag for item in prepared_jobs)
            image_result_events.extend(
                ImageResultEvent(delivery_tag=item.job.delivery_tag, image_id=item.job.image_id, status="indexed")
                for item in prepared_jobs
            )
            return IngestionOutcome(acked, requeue, dead_letter, image_result_events)
        except Exception:
            requeue.extend(item.job.delivery_tag for item in prepared_jobs)
            return IngestionOutcome(acked, requeue, dead_letter, image_result_events)
        finally:
            for path in temp_paths_to_cleanup:
                path.unlink(missing_ok=True)
