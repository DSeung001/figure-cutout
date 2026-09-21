from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

from figure_cutout.ml.factory import build_local_placeholder_pipeline


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * ratio), len(ordered) - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the local cutout pipeline.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, default=Path("data/benchmark-results"))
    args = parser.parse_args()

    images = sorted(
        path for path in args.input_dir.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        raise SystemExit(f"No supported images found under {args.input_dir}")

    pipeline = build_local_placeholder_pipeline()
    latencies_ms: list[float] = []
    failures: list[dict[str, str]] = []

    started = time.perf_counter()
    for index, source in enumerate(images):
        output = args.result_dir / f"{index:06d}-{source.stem}.png"
        item_started = time.perf_counter()
        try:
            pipeline.run(source, output)
            latencies_ms.append((time.perf_counter() - item_started) * 1000)
        except Exception as exc:
            failures.append({"source": str(source), "error": str(exc)})

    elapsed = time.perf_counter() - started
    success_count = len(latencies_ms)

    report = {
        "pipeline_version": pipeline.version,
        "pipeline": {
            "detector": pipeline.detector.name,
            "segmenter": pipeline.segmenter.name,
            "refiner": pipeline.refiner.name,
            "quality_evaluator": pipeline.quality_evaluator.name,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "input_dir": str(args.input_dir),
        "image_count": len(images),
        "success_count": success_count,
        "failure_count": len(failures),
        "elapsed_seconds": elapsed,
        "throughput_images_per_second": success_count / elapsed if elapsed else 0.0,
        "latency_ms": {
            "mean": statistics.fmean(latencies_ms) if latencies_ms else 0.0,
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
        },
        "failures": failures,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
