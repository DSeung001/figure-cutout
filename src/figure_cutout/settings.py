from __future__ import annotations

import os
from pathlib import Path

FIGURE_PROJECT_ENV = "FIGURE_PROJECT_DIR"
DEFAULT_FIGURE_PROJECT_DIR = Path.home() / "figure_project"


def figure_project_dir() -> Path:
    """Folder shared with subculture-researcher; `FIGURE_PROJECT_DIR` overrides ~/figure_project."""
    value = os.environ.get(FIGURE_PROJECT_ENV, "").strip()
    return Path(value).expanduser() if value else DEFAULT_FIGURE_PROJECT_DIR


def export_dir() -> Path:
    """Image exports written by subculture-researcher (`<stamp>/` folders)."""
    return figure_project_dir() / "exports"


def latest_export() -> Path:
    """Newest complete export; stamps are UTC timestamps, so name order is chronological."""
    root = export_dir()
    exports = sorted(
        path for path in root.glob("*") if path.is_dir() and (path / "index.json").is_file()
    )
    if not exports:
        raise FileNotFoundError(
            f"No complete image export under {root}. Export images from subculture-researcher "
            f"or set {FIGURE_PROJECT_ENV} in .env."
        )
    return exports[-1]
