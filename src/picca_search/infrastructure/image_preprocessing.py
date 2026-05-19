from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageOps


@contextmanager
def prepare_inference_image(image_path: Path) -> Iterator[Path]:
    with Image.open(image_path) as image:
        normalized_image = ImageOps.exif_transpose(image)
        width, height = normalized_image.size
        long_edge = max(width, height)
        needs_orientation_normalization = _needs_orientation_normalization(image)
        if long_edge <= 2048 and not needs_orientation_normalization:
            yield image_path
            return

        output_image = normalized_image
        if long_edge > 2048:
            scale = 2048 / long_edge
            resized_size = (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            )
            output_image = normalized_image.resize(resized_size, Image.Resampling.LANCZOS)

        suffix, output_format = _select_output_format(image_path, normalized_image)

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)

        try:
            if output_format == "PNG":
                output_image.save(temporary_path, format=output_format)
            else:
                output_image.convert("RGB").save(
                    temporary_path,
                    format=output_format,
                    quality=90,
                    optimize=True,
                )
            yield temporary_path
        finally:
            temporary_path.unlink(missing_ok=True)


def _needs_orientation_normalization(image: Image.Image) -> bool:
    orientation = image.getexif().get(274, 1)
    return orientation != 1


def _select_output_format(image_path: Path, image: Image.Image) -> tuple[str, str]:
    has_alpha = "A" in image.getbands() or image.info.get("transparency") is not None
    if has_alpha or image_path.suffix.lower() in {".png", ".bmp"}:
        return ".png", "PNG"
    return ".jpg", "JPEG"
