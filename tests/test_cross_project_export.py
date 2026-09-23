"""Optional end-to-end contract check against a separately installed exporter.

Run with SUBCULTURE_RESEARCHER_DIR pointing to its checkout (with .venv).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from figure_cutout.benchmark import benchmark_pipeline
from figure_cutout.compare import compare_runs
from figure_cutout.dataset import collect_samples, init_export_dataset


def test_export_to_compare_across_projects(tmp_path, monkeypatch):
    checkout = os.environ.get("SUBCULTURE_RESEARCHER_DIR")
    if not checkout:
        pytest.skip("Set SUBCULTURE_RESEARCHER_DIR for cross-project integration")
    monkeypatch.setenv("FIGURE_PROJECT_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    producer = Path(checkout)
    fixture = Path(__file__).parent / "fixtures/image-export-v4.json"
    assert fixture.read_bytes() == (producer / "tests/fixtures/image-export-v4.json").read_bytes()
    script = """
import base64, json, os, shutil, zipfile
from pathlib import Path
from unittest.mock import Mock
from subculture.library.application.export_images import export_images, ExportOptions, zip_export
root = Path(os.environ["FIGURE_PROJECT_DIR"])
fixture = json.loads(Path("tests/fixtures/image-export-v4.json").read_text())
entry = fixture["entry"]
library = Mock()
library.items_by_id.return_value = {entry["id"]: {**entry, "_id": entry["id"]}}
first = export_images(library, [entry["id"]], directory=root / "exports/first",
    fetch=lambda url, **kwargs: (base64.b64decode(fixture["image_base64"]), "image/png"),
    options=ExportOptions(pause_seconds=0))
second = export_images(library, [entry["id"]], directory=root / "exports/second",
    fetch=Mock(side_effect=AssertionError("No network on reuse")))
shutil.rmtree(first)
with zipfile.ZipFile(zip_export(second)) as archive:
    archive.extractall(root / "portable")
"""
    subprocess.run([str(producer / ".venv/bin/python"), "-c", script], cwd=producer, check=True)
    root = tmp_path / "exports/second"
    assert init_export_dataset(root)["accepted_count"] == 1
    key = collect_samples(root)[0].id
    metadata = root / "metadata" / f"{key}.json"
    data = json.loads(metadata.read_text())
    data["tags"] = ["display_base"]
    metadata.write_text(json.dumps(data))
    split = (root / "splits/val.txt").read_bytes()
    assert init_export_dataset(root)["metadata_created"] == 0
    assert json.loads(metadata.read_text())["tags"] == ["display_base"]
    assert (root / "splits/val.txt").read_bytes() == split
    report = benchmark_pipeline("placeholder", root)
    assert report["success_count"] == 1 and report["failure_count"] == 0
    comparison = compare_runs(root, ["placeholder"])
    assert comparison["samples"][0]["sample_id"] == key
    # Extracted ZIP resolves with no shared storage configured.
    monkeypatch.setenv("FIGURE_PROJECT_DIR", str(tmp_path / "unavailable"))
    assert init_export_dataset(tmp_path / "portable")["accepted_count"] == 1
    assert collect_samples(tmp_path / "portable")[0].path.is_file()
