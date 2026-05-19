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
class IngestionOutcome:
    acked_delivery_tags: list[int]
    requeue_delivery_tags: list[int]
    dead_letter_delivery_tags: list[int]


@dataclass(frozen=True)
class PendingImageJob:
    delivery_tag: int
    image_id: str


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
        documents: list[ImageDocument] = []
        dense_paths: list[Path] = []
        document_paths: list[Path] = []
        florence_texts: list[str] = []
        ocr_texts_for_sparse: list[str] = []
        temp_paths_to_cleanup: list[Path] = []
        successful_jobs: list[PendingImageJob] = []
        extracted_texts: list[ExtractedImageText] = []

        try:
            for job in jobs:
                local_path = self.storage.download_to_tempfile(job.image_id)
                temp_paths_to_cleanup.append(local_path)
                with prepare_inference_image(local_path) as inference_path:
                    ocr_text = self.ocr_client.extract_text(inference_path)
                    caption = self.caption_client.caption(inference_path)
                    extracted = ExtractedImageText.create(ocr_text=ocr_text, caption=caption)
                    extracted_texts.append(extracted)
                    successful_jobs.append(job)
                    document_paths.append(local_path)
                    if inference_path == local_path:
                        dense_paths.append(local_path)
                    else:
                        with tempfile.NamedTemporaryFile(
                            suffix=inference_path.suffix,
                            delete=False,
                        ) as temporary_file:
                            retained_path = Path(temporary_file.name)
                        shutil.copy2(inference_path, retained_path)
                        temp_paths_to_cleanup.append(retained_path)
                        dense_paths.append(retained_path)
                    florence_texts.append(extracted.caption or extracted.combined)
                    if extracted.ocr_text != "":
                        ocr_texts_for_sparse.append(extracted.ocr_text)
            if not successful_jobs:
                return IngestionOutcome(acked, requeue, dead_letter)

            dense_vectors = self.dense_client.encode_images(dense_paths)
            florence_sparse_vectors = self.sparse_client.encode_texts(florence_texts)
            ocr_sparse_vectors = self.sparse_client.encode_texts(ocr_texts_for_sparse)
            next_ocr_sparse = iter(ocr_sparse_vectors)
            for job, local_path, extracted, dense_vector, florence_sparse in zip(
                successful_jobs,
                document_paths,
                extracted_texts,
                dense_vectors,
                florence_sparse_vectors,
                strict=True,
            ):
                documents.append(
                    ImageDocument.create(
                        image_id=ImageId.from_object_key(job.image_id),
                        image_path=ImagePath.create(local_path),
                        source_path=ImageSourcePath.create(self.storage.uri_for(job.image_id)),
                        dense_vector=dense_vector,
                        florence_sparse_vector=florence_sparse,
                        text=extracted.combined,
                        ocr_sparse_vector=next(next_ocr_sparse) if extracted.ocr_text != "" else None,
                        ocr_text=extracted.ocr_text,
                        caption=extracted.caption,
                    )
                )

            self.index.upsert(documents)
            acked.extend(job.delivery_tag for job in successful_jobs)
            return IngestionOutcome(acked, requeue, dead_letter)
        except (OSError, ValueError):
            dead_letter.extend(job.delivery_tag for job in jobs)
            return IngestionOutcome(acked, requeue, dead_letter)
        except Exception:
            requeue.extend(job.delivery_tag for job in jobs)
            return IngestionOutcome(acked, requeue, dead_letter)
        finally:
            for path in temp_paths_to_cleanup:
                path.unlink(missing_ok=True)
