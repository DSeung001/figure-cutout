from __future__ import annotations

import csv
from pathlib import Path

import pytest

from figure_cutout.benchmark import benchmark_pipeline
from figure_cutout.compare import compare_runs
from figure_cutout.dataset import prepare_eval_set


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
