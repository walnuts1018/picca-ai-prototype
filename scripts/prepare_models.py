#!/usr/bin/env python3
import argparse
import logging
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from optimum.exporters.onnx import main_export


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Model constants
SIGLIP_MODEL = "llm-jp/waon-siglip2-base-patch16-256"
SPLADE_MODEL = "bizreach-inc/light-splade-japanese-28M"
CAT_TRANSLATE_MODEL = "cyberagent/CAT-Translate-0.8b"
FLORENCE2_MODEL = "microsoft/Florence-2-base-ft"

# For PaddleOCR, we'll try to trigger a download by initializing the classes
# Note: PaddleOCR models are typically managed by PaddlePaddle's own downloader.
# We'll document the cache directory for mounting.


def export_hf_to_onnx(model_id: str, output_dir: Path, task: str = "feature-extraction") -> None:
    logger.info(f"Exporting {model_id} to ONNX (task: {task})...")
    output_path = output_dir / model_id.split("/")[-1]
    
    # Clean up if exists
    if output_path.exists():
        logger.info(f"Removing existing directory: {output_path}")
        shutil.rmtree(output_path)
    
    main_export(
        model_name_or_path=model_id,
        output=output_path,
        task=task,
        # opaquely handle trust_remote_code if needed
    )
    logger.info(f"Successfully exported {model_id} to {output_path}")


def download_hf_model(model_id: str, output_dir: Path) -> None:
    logger.info(f"Downloading {model_id} PyTorch weights...")
    output_path = output_dir / model_id.split("/")[-1]
    
    snapshot_download(
        repo_id=model_id,
        local_dir=output_path,
        local_dir_use_symlinks=False,
    )
    logger.info(f"Successfully downloaded {model_id} to {output_path}")


def prepare_paddleocr(output_dir: Path) -> None:
    logger.info("Triggering PaddleOCR model downloads...")
    paddlex_home = output_dir / "paddlex"
    os.environ["PADDLEX_HOME"] = str(paddlex_home.absolute())
    
    try:
        from paddleocr import PaddleOCRVL, TextDetection
        
        logger.info(f"Initializing TextDetection model...")
        TextDetection(model_name="PP-OCRv5_mobile_det")
        
        logger.info(f"Initializing PaddleOCRVL pipeline...")
        PaddleOCRVL(pipeline_version="v1")
        
        # Manually copy from default cache if not in paddlex_home
        default_paddlex_home = Path.home() / ".paddlex"
        if default_paddlex_home.exists() and not (paddlex_home / "official_models").exists():
            logger.info(f"Copying models from {default_paddlex_home} to {paddlex_home}...")
            paddlex_home.mkdir(parents=True, exist_ok=True)
            # Copy the official_models directory
            src = default_paddlex_home / "official_models"
            dst = paddlex_home / "official_models"
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

        logger.info(f"PaddleOCR models prepared in {paddlex_home}")
        logger.info("Hint: To mount these in Docker, ensure ./models is mounted to /models and PADDLEX_HOME=/models/paddlex is set.")
    except ImportError:
        logger.warning("paddleocr not installed, skipping PaddleOCR model preparation.")


def main():
    parser = argparse.ArgumentParser(description="Prepare models by exporting to ONNX or downloading weights locally.")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Directory to save models")
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX export")
    parser.add_argument("--skip-download", action="store_true", help="Skip PyTorch weights download")
    parser.add_argument("--skip-paddle", action="store_true", help="Skip PaddleOCR preparation")
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.skip_onnx:
        # SigLIP and SPLADE are feature-extraction tasks
        export_hf_to_onnx(SIGLIP_MODEL, args.output_dir, task="feature-extraction")
        export_hf_to_onnx(SPLADE_MODEL, args.output_dir, task="feature-extraction")
        
        # CAT-Translate is a Causal LM
        export_hf_to_onnx(CAT_TRANSLATE_MODEL, args.output_dir, task="text-generation-with-past")
        
    if not args.skip_download:
        # Florence-2 (Not natively supported by Optimum CLI yet without custom scripts)
        download_hf_model(FLORENCE2_MODEL, args.output_dir)
        
    if not args.skip_paddle:
        prepare_paddleocr(args.output_dir)


if __name__ == "__main__":
    main()
