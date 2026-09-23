from __future__ import annotations

import json
import platform
import statistics
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from figure_cutout.dataset import Sample, collect_samples, dataset_name
from figure_cutout.domain.models import CutoutResult
from figure_cutout.ml.factory import build_pipeline, list_pipelines
from figure_cutout.ml.pipeline import FigureCutoutPipeline


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * ratio), len(ordered) - 1)
    return ordered[index]


def detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def peak_vram_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except ImportError:
        return None


def write_debug_artifacts(sample_dir: Path, result: CutoutResult) -> None:
    trace = result.trace
    if trace is None:
        return
    sample_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        sample_dir / "detection.json",
        {
            "bbox": list(trace.detection.bbox),
            "score": trace.detection.score,
            "label": trace.detection.label,
        },
    )
    Image.fromarray(np.asarray(trace.raw_mask, dtype=np.uint8)).save(sample_dir / "raw-mask.png")
    Image.fromarray(np.asarray(trace.refined_mask, dtype=np.uint8)).save(
        sample_dir / "refined-mask.png"
    )
    _write_json(
        sample_dir / "metrics.json",
        {
            "quality": asdict(result.quality),
            "metadata": result.metadata,
            "output": str(result.output),
        },
    )


def write_failure_artifact(sample_dir: Path, sample: Sample, exc: Exception) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        sample_dir / "error.json",
        {
            "sample_id": sample.id,
            "source": str(sample.path),
            "type": type(exc).__name__,
            "error": str(exc),
        },
    )


def build_report(
    *,
    pipeline_id: str,
    pipeline: FigureCutoutPipeline,
    dataset: Path,
    split: str,
    image_count: int,
    latencies_ms: list[float],
    failures: list[dict[str, str]],
    elapsed: float,
    run_id: str,
    device: str,
    result_dir: Path,
    debug_dir: Path | None,
) -> dict[str, Any]:
    success_count = len(latencies_ms)
    report: dict[str, Any] = {
        "run_id": run_id,
        "pipeline_id": pipeline_id,
        "pipeline_version": pipeline.version,
        "model": {
            "detector": pipeline.detector.name,
            "segmenter": pipeline.segmenter.name,
            "refiner": pipeline.refiner.name,
            "quality_evaluator": pipeline.quality_evaluator.name,
        },
        "device": device,
        "dataset": dataset_name(dataset),
        "dataset_path": str(dataset),
        "split": split,
        "image_count": image_count,
        "success_count": success_count,
        "failure_count": len(failures),
        "elapsed_seconds": elapsed,
        "throughput_images_per_second": success_count / elapsed if elapsed else 0.0,
        "latency_ms": {
            "mean": statistics.fmean(latencies_ms) if latencies_ms else 0.0,
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "result_dir": str(result_dir),
        "debug_dir": str(debug_dir) if debug_dir else None,
        "failures": failures,
        "created_at": datetime.now(UTC).isoformat(),
    }
    vram = peak_vram_mb()
    if vram is not None:
        report["peak_vram_mb"] = vram
    return report


def benchmark_pipeline(
    pipeline_id: str,
    dataset: Path,
    *,
    split: str = "val",
    result_root: Path = Path("data/benchmark-results"),
    benchmark_root: Path = Path("benchmarks"),
    debug_root: Path = Path("data/debug"),
    write_debug: bool = True,
) -> dict[str, Any]:
    samples = collect_samples(dataset, split=split)
    pipeline = build_pipeline(pipeline_id)
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"

    result_dir = result_root / pipeline_id / run_id
    debug_dir = debug_root / run_id if write_debug else None

    latencies_ms: list[float] = []
    failures: list[dict[str, str]] = []

    started = time.perf_counter()
    for index, sample in enumerate(samples):
        output = result_dir / f"{index:06d}-{sample.id}.png"
        item_started = time.perf_counter()
        try:
            result = pipeline.run(sample.path, output)
        except Exception as exc:  # noqa: BLE001 - per-image failures are recorded, not fatal
            failures.append({"sample_id": sample.id, "source": str(sample.path), "error": str(exc)})
            if debug_dir is not None:
                write_failure_artifact(debug_dir / output.stem, sample, exc)
            continue
        latencies_ms.append((time.perf_counter() - item_started) * 1000)
        if debug_dir is not None:
            write_debug_artifacts(debug_dir / output.stem, result)
    elapsed = time.perf_counter() - started

    report = build_report(
        pipeline_id=pipeline_id,
        pipeline=pipeline,
        dataset=dataset,
        split=split,
        image_count=len(samples),
        latencies_ms=latencies_ms,
        failures=failures,
        elapsed=elapsed,
        run_id=run_id,
        device=detect_device(),
        result_dir=result_dir,
        debug_dir=debug_dir,
    )

    output_path = benchmark_root / pipeline_id / f"{run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, report)
    report["benchmark_path"] = str(output_path)
    return report


def _skipped_report(pipeline_id: str, error: ImportError) -> dict[str, Any]:
    return {
        "run_id": None,
        "pipeline_id": pipeline_id,
        "skipped": True,
        "success_count": 0,
        "failure_count": 0,
        "device": detect_device(),
        "latency_ms": {"p50": 0.0},
        "benchmark_path": None,
        "error": str(error),
    }


def eval_pipelines(
    dataset: Path,
    pipeline_ids: list[str] | None = None,
    *,
    split: str = "val",
    write_debug: bool = True,
) -> list[dict[str, Any]]:
    """Benchmark pipelines on one split. Pipelines with missing optional deps are skipped."""
    selected = pipeline_ids or list_pipelines()
    if not selected:
        raise ValueError("No pipelines registered.")

    reports: list[dict[str, Any]] = []
    for pipeline_id in selected:
        try:
            report = benchmark_pipeline(pipeline_id, dataset, split=split, write_debug=write_debug)
        except ImportError as exc:
            report = _skipped_report(pipeline_id, exc)
        reports.append(report)
    return reports
