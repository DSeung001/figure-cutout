from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def load_rgba(path: Path) -> Image.Image:
    """Decode an input image as displayed: first frame, EXIF orientation applied, RGBA."""
    with Image.open(path) as image:
        image.seek(0)  # animated GIF/WebP: first frame only
        return ImageOps.exif_transpose(image).convert("RGBA")
