from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from figure_cutout.benchmark import benchmark_pipeline, eval_pipelines
from figure_cutout.cli import main
from figure_cutout.dataset import collect_images, prepare_eval_set
from figure_cutout.domain.models import Detection
from figure_cutout.ml import factory as factory_module
from figure_cutout.ml import rembg_adapter
from figure_cutout.ml.factory import build_pipeline, list_pipelines


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return prepare_eval_set(tmp_path / "figure-v1", count=3, seed=1)


def test_list_pipelines_includes_placeholder_and_rembg() -> None:
    assert {"placeholder", "rembg"} <= set(list_pipelines())


def test_build_unknown_pipeline_raises() -> None:
    with pytest.raises(ValueError, match="Unknown pipeline"):
        build_pipeline("nope")


def test_pipeline_run_returns_trace(dataset: Path, tmp_path: Path) -> None:
    source = collect_images(dataset)[0]
    result = build_pipeline("placeholder").run(source, tmp_path / "out.png")
    assert result.trace is not None
    assert result.trace.detection.label == "figure"
    assert np.asarray(result.trace.raw_mask).shape == np.asarray(result.trace.refined_mask).shape


def test_prepare_eval_set_metadata(dataset: Path) -> None:
    meta = json.loads((dataset / "metadata" / "sample-001.json").read_text(encoding="utf-8"))
    assert meta["id"] == "sample-001"
    assert isinstance(meta["tags"], list)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"] == "figure-v1"
    assert manifest["seed"] == 1


def test_prepare_eval_set_refuses_overwrite(dataset: Path) -> None:
    with pytest.raises(FileExistsError):
        prepare_eval_set(dataset, count=3)
    prepare_eval_set(dataset, count=3, overwrite=True)


def test_collect_images_rejects_missing_split_entry(dataset: Path) -> None:
    val = dataset / "splits" / "val.txt"
    val.write_text(val.read_text(encoding="utf-8") + "ghost.jpg\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="ghost.jpg"):
        collect_images(dataset, split="val")


def test_collect_images_rejects_empty_split(dataset: Path) -> None:
    with pytest.raises(FileNotFoundError):
        collect_images(dataset, split="train")


def test_benchmark_writes_report_and_debug(dataset: Path, tmp_path: Path) -> None:
    report = benchmark_pipeline(
        "placeholder",
        dataset,
        result_root=tmp_path / "results",
        benchmark_root=tmp_path / "benchmarks",
        debug_root=tmp_path / "debug",
    )
    assert report["pipeline_id"] == "placeholder"
    assert report["dataset"] == "figure-v1"
    assert report["split"] == "val"
    assert (report["image_count"], report["success_count"], report["failure_count"]) == (3, 3, 0)
    assert "device" in report
    saved = json.loads(Path(report["benchmark_path"]).read_text(encoding="utf-8"))
    assert saved["run_id"] == report["run_id"]

    sample_dirs = sorted(Path(report["debug_dir"]).iterdir())
    assert len(sample_dirs) == 3
    for name in ("detection.json", "raw-mask.png", "refined-mask.png", "metrics.json"):
        assert (sample_dirs[0] / name).is_file()


def test_benchmark_records_failure_artifact(dataset: Path, tmp_path: Path) -> None:
    (dataset / "images" / "sample-002.jpg").write_bytes(b"not an image")
    report = benchmark_pipeline(
        "placeholder",
        dataset,
        result_root=tmp_path / "results",
        benchmark_root=tmp_path / "benchmarks",
        debug_root=tmp_path / "debug",
    )
    assert (report["success_count"], report["failure_count"]) == (2, 1)
    errors = list(Path(report["debug_dir"]).glob("*/error.json"))
    assert len(errors) == 1
    assert json.loads(errors[0].read_text(encoding="utf-8"))["source"].endswith("sample-002.jpg")


def test_eval_skips_pipeline_with_missing_dependency(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _broken() -> object:
        raise ImportError("rembg missing")

    monkeypatch.setitem(factory_module.PIPELINE_REGISTRY, "rembg", _broken)
    monkeypatch.chdir(tmp_path)
    reports = eval_pipelines(dataset, pipeline_ids=["placeholder", "rembg"], write_debug=False)
    assert reports[0]["success_count"] == 3
    assert reports[1]["skipped"] is True


def test_cli_eval_exit_codes(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _broken() -> object:
        raise ImportError("rembg missing")

    monkeypatch.setitem(factory_module.PIPELINE_REGISTRY, "rembg", _broken)
    monkeypatch.chdir(tmp_path)
    base = ["eval", "--dataset", str(dataset), "--no-debug"]
    assert main([*base, "--pipeline", "placeholder"]) == 0
    assert main(base) == 0  # implicit selection tolerates skipped pipelines
    assert main([*base, "--pipeline", "rembg"]) == 1  # explicit selection does not


def test_rembg_segmenter_uses_alpha_and_bbox(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_remove(image: Image.Image, session: object) -> Image.Image:
        alpha = Image.new("L", image.size, 255)
        return Image.merge("RGBA", (*image.convert("RGB").split(), alpha))

    def fake_new_session(model_name: str) -> object:
        calls.append(model_name)
        return object()

    monkeypatch.setattr(rembg_adapter, "_load_rembg", lambda: (fake_remove, fake_new_session))
    segmenter = rembg_adapter.RembgSegmenter("tiny")
    image = np.zeros((10, 20, 4), dtype=np.uint8)

    full = segmenter.segment(image, Detection(bbox=(0, 0, 20, 10), score=1.0))
    assert full.mask.shape == (10, 20) and full.confidence == 1.0
    cropped = segmenter.segment(image, Detection(bbox=(0, 0, 10, 10), score=1.0))
    assert np.count_nonzero(cropped.mask) == 100
    assert segmenter.name == "rembg-tiny"
    assert calls == ["tiny"]  # session created once and reused


def test_rembg_quality_evaluator_reasons() -> None:
    evaluator = rembg_adapter.RembgQualityEvaluator()
    detection = Detection(bbox=(0, 0, 2, 2), score=1.0)
    ok = evaluator.evaluate(np.zeros((2, 2)), detection, np.full((2, 2), 255), 0.6)
    assert not ok.requires_review and ok.reasons == []
    empty = evaluator.evaluate(np.zeros((2, 2)), detection, np.zeros((2, 2)), 0.0)
    assert empty.requires_review and empty.reasons == ["empty_mask"]
    low = evaluator.evaluate(np.zeros((2, 2)), detection, np.full((2, 2), 255), 0.1)
    assert low.requires_review and low.reasons == ["low_foreground_ratio"]
