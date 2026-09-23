from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from figure_cutout.dataset import collect_images, dataset_name

TILE_HEIGHT = 512
LABEL_HEIGHT = 28
CHECKER_SIZE = 16
REVIEW_COLUMNS = ["sample_id", "pipeline_id", "score_1to5", "failure_tags", "note"]


def latest_run(
    pipeline_id: str,
    dataset: str,
    split: str,
    benchmark_root: Path = Path("benchmarks"),
) -> dict[str, Any]:
    """Most recent benchmark report of a pipeline on the given dataset/split."""
    # run_id starts with a UTC timestamp, so name order is chronological.
    for path in sorted((benchmark_root / pipeline_id).glob("*.json"), reverse=True):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("dataset") == dataset and report.get("split") == split:
            report["benchmark_path"] = str(path)
            return report
    raise FileNotFoundError(
        f"No benchmark run for pipeline '{pipeline_id}' on {dataset}/{split}. "
        f"Run: figure-cutout eval --dataset <path> --pipeline {pipeline_id}"
    )


def _sample_outputs(report: dict[str, Any]) -> dict[str, tuple[Path, bool | None]]:
    """Map sample stem -> (result image, requires_review) for one run."""
    result_dir = Path(report["result_dir"])
    debug_dir = Path(report["debug_dir"]) if report.get("debug_dir") else None
    outputs: dict[str, tuple[Path, bool | None]] = {}
    for path in sorted(result_dir.glob("*.png")):
        # Result files are named "<index>-<sample stem>.png" by the benchmark.
        sample_stem = path.stem.split("-", 1)[1]
        review: bool | None = None
        metrics = debug_dir / path.stem / "metrics.json" if debug_dir else None
        if metrics is not None and metrics.is_file():
            review = json.loads(metrics.read_text(encoding="utf-8"))["quality"]["requires_review"]
        outputs[sample_stem] = (path, review)
    return outputs


def _checkerboard(size: tuple[int, int]) -> Image.Image:
    board = Image.new("RGBA", size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(board)
    for y in range(0, size[1], CHECKER_SIZE):
        for x in range(0, size[0], CHECKER_SIZE):
            if (x // CHECKER_SIZE + y // CHECKER_SIZE) % 2:
                draw.rectangle(
                    (x, y, x + CHECKER_SIZE - 1, y + CHECKER_SIZE - 1), fill=(204, 204, 204, 255)
                )
    return board


def _tile(image: Image.Image, label: str, flagged: bool) -> Image.Image:
    width = max(1, round(image.width * TILE_HEIGHT / image.height))
    resized = image.convert("RGBA").resize((width, TILE_HEIGHT), Image.Resampling.LANCZOS)
    body = _checkerboard(resized.size)
    body.alpha_composite(resized)

    tile = Image.new("RGBA", (width, TILE_HEIGHT + LABEL_HEIGHT), (255, 255, 255, 255))
    draw = ImageDraw.Draw(tile)
    draw.rectangle(
        (0, 0, width, LABEL_HEIGHT), fill=(200, 60, 60, 255) if flagged else (40, 40, 40, 255)
    )
    try:
        font: Any = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((6, 5), label + ("  REVIEW" if flagged else ""), fill=(255, 255, 255, 255), font=font)
    tile.alpha_composite(body, (0, LABEL_HEIGHT))
    return tile


def _grid(tiles: list[Image.Image]) -> Image.Image:
    gap = 8
    width = sum(tile.width for tile in tiles) + gap * (len(tiles) - 1)
    grid = Image.new("RGBA", (width, tiles[0].height), (255, 255, 255, 255))
    x = 0
    for tile in tiles:
        grid.alpha_composite(tile, (x, 0))
        x += tile.width + gap
    return grid


def compare_runs(
    dataset: Path,
    pipeline_ids: list[str],
    *,
    split: str = "val",
    benchmark_root: Path = Path("benchmarks"),
    compare_root: Path = Path("data/compare"),
) -> dict[str, Any]:
    """Build per-sample side-by-side sheets from the latest run of each pipeline."""
    if not pipeline_ids:
        raise ValueError("At least one pipeline id is required.")
    name = dataset_name(dataset)
    reports = [latest_run(pid, name, split, benchmark_root) for pid in pipeline_ids]
    outputs = {report["pipeline_id"]: _sample_outputs(report) for report in reports}

    compare_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = compare_root / name / compare_id
    sheets_dir = out_dir / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    for source in collect_images(dataset, split=split):
        tiles = [_tile(Image.open(source), "original", False)]
        row: dict[str, Any] = {"sample_id": source.stem, "outputs": {}, "requires_review": {}}
        for pid in pipeline_ids:
            result = outputs[pid].get(source.stem)
            if result is None:
                row["outputs"][pid] = None
                continue
            path, review = result
            tiles.append(_tile(Image.open(path), pid, bool(review)))
            row["outputs"][pid] = str(path)
            row["requires_review"][pid] = review
        sheet = sheets_dir / f"{source.stem}.png"
        _grid(tiles).convert("RGB").save(sheet)
        row["sheet"] = str(sheet)
        samples.append(row)

    review_csv = out_dir / "review.csv"
    with review_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(REVIEW_COLUMNS)
        for row in samples:
            for pid in pipeline_ids:
                writer.writerow([row["sample_id"], pid, "", "", ""])

    summary = {
        "compare_id": compare_id,
        "dataset": name,
        "split": split,
        "pipelines": [
            {
                "pipeline_id": report["pipeline_id"],
                "run_id": report["run_id"],
                "segmenter": report["model"]["segmenter"],
                "benchmark_path": report["benchmark_path"],
                "success_count": report["success_count"],
                "failure_count": report["failure_count"],
                "latency_ms": report["latency_ms"],
                "review_count": sum(
                    1 for row in samples if row["requires_review"].get(report["pipeline_id"])
                ),
            }
            for report in reports
        ],
        "samples": samples,
        "review_csv": str(review_csv),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary["output_dir"] = str(out_dir)
    return summary
