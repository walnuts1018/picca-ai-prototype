#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Force PADDLEX_HOME before ANY other imports
def setup_paddlex_env():
    # If explicitly passed via CLI, we'll handle it in main, 
    # but for module-level initialization we need a default or a check.
    # Here we default to models/paddlex relative to the script's project root.
    project_root = Path(__file__).parent.parent.absolute()
    paddlex_home = project_root / "models" / "paddlex"
    paddlex_home.mkdir(parents=True, exist_ok=True)
    os.environ["PADDLEX_HOME"] = str(paddlex_home)
    os.environ["PADDLE_HOME"] = str(paddlex_home)
    os.environ["PADDLE_PDX_HOME"] = str(paddlex_home)

setup_paddlex_env()

import argparse
import logging
import shutil
import subprocess

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


def export_paddle_to_onnx(model_dir: Path) -> None:
    """Convert Paddle inference models to ONNX using paddle2onnx CLI."""
    # Check for Paddle 3.0+ json format first, then traditional pdmodel
    model_file = model_dir / "inference.json"
    if not model_file.exists():
        model_file = model_dir / "inference.pdmodel"

    params_file = model_dir / "inference.pdiparams"
    output_file = model_dir / "inference.onnx"

    if not model_file.exists() or not params_file.exists():
        logger.debug(f"Skipping {model_dir}: required Paddle files not found.")
        return

    logger.info(f"Converting Paddle model in {model_dir} to ONNX...")
    try:
        cmd = [
            sys.executable, "-m", "paddle2onnx.command",
            "--model_dir", str(model_dir),
            "--model_filename", model_file.name,
            "--params_filename", params_file.name,
            "--save_file", str(output_file),
            "--opset_version", "11",
            "--enable_onnx_checker", "True"
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Successfully exported to {output_file}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to convert {model_dir} to ONNX: {e.stderr}")
    except Exception as e:
        logger.error(f"Error during conversion of {model_dir}: {e}")


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
    # Use the globally set PADDLEX_HOME or update it if output_dir is different
    paddlex_home = Path(os.environ["PADDLEX_HOME"])
    
    try:
        # Install HPI dependencies if needed (for ONNX Runtime support)
        logger.info("Installing HPI dependencies...")
        try:
            subprocess.run(["paddlex", "--install", "hpi-gpu"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            logger.warning("Failed to install hpi-gpu, trying hpi-cpu...")
            subprocess.run(["paddlex", "--install", "hpi-cpu"], check=False)
        
        from paddleocr import PaddleOCRVL, TextDetection
        
        logger.info(f"Initializing TextDetection model...")
        TextDetection(model_name="PP-OCRv5_mobile_det")
        
        logger.info(f"Initializing PaddleOCRVL pipeline...")
        PaddleOCRVL(pipeline_version="v1")

        # Fallback: if initialization used ~/.paddlex despite env var, copy it to models/paddlex
        default_paddlex_home = Path.home() / ".paddlex"
        if default_paddlex_home.exists() and not (paddlex_home / "official_models").exists():
            logger.info(f"Copying models from {default_paddlex_home} to {paddlex_home}...")
            shutil.copytree(default_paddlex_home / "official_models", paddlex_home / "official_models", dirs_exist_ok=True)

        logger.info(f"PaddleOCR models prepared in {paddlex_home}")

        # Export all official Paddle models to ONNX
        official_models_dir = paddlex_home / "official_models"
        if official_models_dir.exists():
            # Process each model directory in official_models
            for model_dir in official_models_dir.iterdir():
                if model_dir.is_dir():
                    export_paddle_to_onnx(model_dir)
                    # Also check for subdirectories (like PaddleOCR-VL/PP-DocLayoutV2)
                    for subdir in model_dir.iterdir():
                        if subdir.is_dir():
                            export_paddle_to_onnx(subdir)

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
    
    # Update PADDLEX_HOME if a custom output-dir was provided
    custom_paddlex_home = (args.output_dir / "paddlex").absolute()
    os.environ["PADDLEX_HOME"] = str(custom_paddlex_home)
    os.environ["PADDLE_HOME"] = str(custom_paddlex_home)
    os.environ["PADDLE_PDX_HOME"] = str(custom_paddlex_home)
    custom_paddlex_home.mkdir(parents=True, exist_ok=True)

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
