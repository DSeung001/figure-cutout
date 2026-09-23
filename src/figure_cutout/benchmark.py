from __future__ import annotations

import json
import platform
import shutil
import statistics
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from figure_cutout.cache import ResultCache, file_sha256
from figure_cutout.dataset import Sample, collect_samples, dataset_name
from figure_cutout.domain.models import CutoutResult
from figure_cutout.ml.factory import build_pipeline, list_pipelines
from figure_cutout.ml.pipeline import FigureCutoutPipeline

DEFAULT_CACHE_ROOT = Path("data/cache")
CACHE_OUTPUT = "output.png"
CACHE_INFO = "cache.json"


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


def store_cached_result(
    cache: ResultCache,
    pipeline_id: str,
    fingerprint: str,
    sha256: str,
    result: CutoutResult,
    latency_ms: float,
) -> None:
    def write(entry: Path) -> None:
        write_debug_artifacts(entry, result)
        shutil.copy2(result.output, entry / CACHE_OUTPUT)
        _write_json(
            entry / CACHE_INFO,
            {
                "pipeline_id": pipeline_id,
                "fingerprint": fingerprint,
                "source_sha256": sha256,
                "source": str(result.source),
                "latency_ms": latency_ms,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )

    cache.store(pipeline_id, fingerprint, sha256, write)


def restore_cached_result(entry: Path, output: Path, debug_sample_dir: Path | None) -> None:
    """Copy a cached result into this run's result and debug directories."""
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(entry / CACHE_OUTPUT, output)
    if debug_sample_dir is None:
        return
    for path in entry.iterdir():
        if path.name not in (CACHE_OUTPUT, CACHE_INFO, "metrics.json"):
            debug_sample_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, debug_sample_dir / path.name)
    metrics_path = entry / "metrics.json"
    if metrics_path.is_file():
        debug_sample_dir.mkdir(parents=True, exist_ok=True)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.update(output=str(output), cache_hit=True, cache_entry=str(entry))
        _write_json(debug_sample_dir / "metrics.json", metrics)


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
    cache_summary: dict[str, Any],
    cache_hits: int,
) -> dict[str, Any]:
    fresh_count = len(latencies_ms)
    success_count = fresh_count + cache_hits
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
        # Latency and throughput cover fresh inference only; cache hits are excluded.
        "throughput_images_per_second": (
            fresh_count / (sum(latencies_ms) / 1000) if latencies_ms else 0.0
        ),
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
        "cache": cache_summary,
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
    cache_root: Path | None = DEFAULT_CACHE_ROOT,
) -> dict[str, Any]:
    """Run a pipeline over a split. With a cache, already processed images skip inference."""
    samples = collect_samples(dataset, split=split)
    pipeline = build_pipeline(pipeline_id)
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"

    result_dir = result_root / pipeline_id / run_id
    debug_dir = debug_root / run_id if write_debug else None

    cache = ResultCache(cache_root) if cache_root is not None else None
    fingerprint = pipeline.fingerprint()

    latencies_ms: list[float] = []
    failures: list[dict[str, str]] = []
    cache_hits = 0

    started = time.perf_counter()
    for index, sample in enumerate(samples):
        output = result_dir / f"{index:06d}-{sample.id}.png"
        debug_sample_dir = debug_dir / output.stem if debug_dir is not None else None
        try:
            sha256 = file_sha256(sample.path) if cache is not None else ""
            entry = cache.lookup(pipeline_id, fingerprint, sha256) if cache is not None else None
            if entry is not None:
                restore_cached_result(entry, output, debug_sample_dir)
                cache_hits += 1
                continue
            item_started = time.perf_counter()
            result = pipeline.run(sample.path, output)
            latency_ms = (time.perf_counter() - item_started) * 1000
        except Exception as exc:  # noqa: BLE001 - per-image failures are recorded, not fatal
            failures.append({"sample_id": sample.id, "source": str(sample.path), "error": str(exc)})
            if debug_sample_dir is not None:
                write_failure_artifact(debug_sample_dir, sample, exc)
            continue
        latencies_ms.append(latency_ms)
        if cache is not None:
            store_cached_result(cache, pipeline_id, fingerprint, sha256, result, latency_ms)
        if debug_sample_dir is not None:
            write_debug_artifacts(debug_sample_dir, result)
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
        cache_summary={
            "enabled": cache is not None,
            "dir": str(cache.run_dir(pipeline_id, fingerprint)) if cache is not None else None,
            "fingerprint": fingerprint,
            "hits": cache_hits,
            "misses": len(latencies_ms),
        },
        cache_hits=cache_hits,
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
        "cache": {"hits": 0, "misses": 0},
        "benchmark_path": None,
        "error": str(error),
    }


def eval_pipelines(
    dataset: Path,
    pipeline_ids: list[str] | None = None,
    *,
    split: str = "val",
    write_debug: bool = True,
    cache_root: Path | None = DEFAULT_CACHE_ROOT,
) -> list[dict[str, Any]]:
    """Benchmark pipelines on one split. Pipelines with missing optional deps are skipped."""
    selected = pipeline_ids or list_pipelines()
    if not selected:
        raise ValueError("No pipelines registered.")

    reports: list[dict[str, Any]] = []
    for pipeline_id in selected:
        try:
            report = benchmark_pipeline(
                pipeline_id,
                dataset,
                split=split,
                write_debug=write_debug,
                cache_root=cache_root,
            )
        except ImportError as exc:
            report = _skipped_report(pipeline_id, exc)
        reports.append(report)
    return reports
