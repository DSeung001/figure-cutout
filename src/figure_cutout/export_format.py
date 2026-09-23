"""Read-only image export v4 validation and storage resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from figure_cutout.settings import figure_project_dir

EXPORT_FORMAT_VERSION = 4


def read_export_index(root: Path) -> list[dict[str, Any]]:
    """index.json of an image export; only formatVersion 4 is accepted."""
    if not (root / "index.json").is_file():
        raise FileNotFoundError(f"No index.json under {root}: export is missing or incomplete.")
    meta_path = root / "export.json"
    version = (
        json.loads(meta_path.read_text(encoding="utf-8")).get("formatVersion")
        if meta_path.is_file()
        else None
    )
    if version != EXPORT_FORMAT_VERSION:
        raise ValueError(
            f"Export formatVersion {version} at {root}; expected {EXPORT_FORMAT_VERSION}. "
            "Re-export the images (see docs/image-export-format.md)."
        )
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    for entry in index:
        for record in entry["files"]:
            if (
                record["status"] in ("ok", "skipped")
                and record.get("storage") not in ("shared", "export")
            ):
                raise ValueError("Unknown image storage")
            key = record["key"]
            if (
                not isinstance(key, str)
                or not key
                or key in (".", "..")
                or "/" in key
                or "\\" in key
            ):
                raise ValueError("Invalid sample key")
    return index


def export_image_path(root: Path, record: dict[str, Any]) -> Path:
    storage = record.get("storage")
    if storage == "shared":
        base, value = figure_project_dir() / "images", record.get("path")
    elif storage == "export":
        base, value = root, record.get("path")
    else:
        raise ValueError(f"Unknown image storage: {storage}")
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("Image path must be relative")
    path = (base / value).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("Image path escapes storage root")
    return path
