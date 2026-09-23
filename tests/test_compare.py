from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from figure_cutout.benchmark import benchmark_pipeline
from figure_cutout.compare import compare_runs
from figure_cutout.dataset import collect_images, init_real_dataset, prepare_eval_set


def test_init_real_dataset_indexes_photos(tmp_path: Path) -> None:
    root = tmp_path / "figure-real-v1"
    (root / "images").mkdir(parents=True)
    for name in ("fig-0002.jpg", "fig-0001.png"):
        Image.new("RGB", (8, 8)).save(root / "images" / name)

    summary = init_real_dataset(root)
    assert summary["image_count"] == 2 and summary["val_split_created"] is True
    assert collect_images(root) == [
        root / "images" / "fig-0001.png",
        root / "images" / "fig-0002.jpg",
    ]
    meta = json.loads((root / "metadata" / "fig-0001.json").read_text(encoding="utf-8"))
    assert meta == {"id": "fig-0001", "tags": [], "difficulty": "unknown", "source": "real"}

    # A later photo is reported, not silently added to the existing split.
    Image.new("RGB", (8, 8)).save(root / "images" / "fig-0003.jpg")
    again = init_real_dataset(root)
    assert again["val_split_created"] is False
    assert again["unlisted_in_val"] == ["fig-0003.jpg"]
    assert again["metadata_created"] == ["fig-0003"]


def test_compare_runs_builds_sheets_and_review_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = prepare_eval_set(tmp_path / "figure-v1", count=2, seed=1)
    benchmark_pipeline("placeholder", dataset)

    summary = compare_runs(dataset, ["placeholder"])
    out_dir = Path(summary["output_dir"])
    assert sorted(p.name for p in (out_dir / "sheets").iterdir()) == [
        "sample-001.png",
        "sample-002.png",
    ]
    assert summary["pipelines"][0]["success_count"] == 2
    with (out_dir / "review.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["sample_id"], r["pipeline_id"]) for r in rows] == [
        ("sample-001", "placeholder"),
        ("sample-002", "placeholder"),
    ]


def test_compare_requires_existing_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = prepare_eval_set(tmp_path / "figure-v1", count=1, seed=1)
    with pytest.raises(FileNotFoundError, match="figure-cutout eval"):
        compare_runs(dataset, ["placeholder"])
