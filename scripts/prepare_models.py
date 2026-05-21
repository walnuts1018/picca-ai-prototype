#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from optimum.exporters.onnx import main_export


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SIGLIP_MODEL = "llm-jp/waon-siglip2-base-patch16-256"
SPLADE_MODEL = "bizreach-inc/light-splade-japanese-28M"
SPLADE_ONNX_EXPORT_TASK = "fill-mask"
CAT_TRANSLATE_MODEL = "cyberagent/CAT-Translate-0.8b"
FLORENCE2_MODEL = "microsoft/Florence-2-base-ft"

PADDLE_TEXT_DETECTION_MODEL = "PP-OCRv5_mobile_det"
PADDLE_LAYOUT_DETECTION_MODEL = "PP-DocLayoutV2"
PADDLE_VL_MODEL = "PaddleOCR-VL"
PADDLE_MODELS = (
    PADDLE_TEXT_DETECTION_MODEL,
    PADDLE_LAYOUT_DETECTION_MODEL,
    PADDLE_VL_MODEL,
)


def export_hf_to_onnx(model_id: str, output_dir: Path, task: str = "feature-extraction") -> None:
    logger.info("Exporting %s to ONNX (task: %s)...", model_id, task)
    output_path = output_dir / model_id.split("/")[-1]

    if output_path.exists():
        logger.info("Removing existing directory: %s", output_path)
        shutil.rmtree(output_path)

    main_export(
        model_name_or_path=model_id,
        output=output_path,
        task=task,
    )
    logger.info("Successfully exported %s to %s", model_id, output_path)


def download_hf_model(model_id: str, output_dir: Path) -> None:
    logger.info("Downloading %s weights...", model_id)
    output_path = output_dir / model_id.split("/")[-1]

    snapshot_download(
        repo_id=model_id,
        local_dir=output_path,
        local_dir_use_symlinks=False,
    )
    logger.info("Successfully downloaded %s to %s", model_id, output_path)


def download_paddlex_model(model_name: str, official_models_dir: Path) -> None:
    output_path = official_models_dir / model_name
    logger.info("Downloading PaddleX model %s to %s...", model_name, output_path)
    snapshot_download(
        repo_id=f"PaddlePaddle/{model_name}",
        local_dir=output_path,
        local_dir_use_symlinks=False,
    )
    logger.info("Successfully downloaded PaddleX model %s", model_name)


def prepare_paddleocr(output_dir: Path) -> None:
    paddlex_home = output_dir / "paddlex"
    official_models_dir = paddlex_home / "official_models"
    official_models_dir.mkdir(parents=True, exist_ok=True)

    for model_name in PADDLE_MODELS:
        download_paddlex_model(model_name, official_models_dir)

    logger.info("PaddleOCR models prepared in %s", paddlex_home)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare local model files so runtime containers do not download models."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Directory to save models")
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX export (SPLADE等のONNXエクスポートをスキップ)")
    parser.add_argument("--skip-download", action="store_true", help="Skip model download")
    parser.add_argument("--skip-paddle", action="store_true", help="Skip PaddleOCR preparation")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # waon-siglip2 は PyTorch weights をそのまま利用するため通常ダウンロード
    if not args.skip_download:
        download_hf_model(SIGLIP_MODEL, args.output_dir)
        download_hf_model(FLORENCE2_MODEL, args.output_dir)

    # SPLADE は ONNX で推論するためエクスポートが必要
    if not args.skip_onnx:
        export_hf_to_onnx(SPLADE_MODEL, args.output_dir, task=SPLADE_ONNX_EXPORT_TASK)
        export_hf_to_onnx(CAT_TRANSLATE_MODEL, args.output_dir, task="text-generation-with-past")

    if not args.skip_paddle:
        prepare_paddleocr(args.output_dir)


if __name__ == "__main__":
    main()
