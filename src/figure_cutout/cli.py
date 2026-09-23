from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from figure_cutout.benchmark import DEFAULT_CACHE_ROOT, benchmark_pipeline, eval_pipelines
from figure_cutout.compare import compare_runs
from figure_cutout.dataset import init_export_dataset, prepare_eval_set
from figure_cutout.ml.factory import DEFAULT_PIPELINE, build_pipeline, list_pipelines
from figure_cutout.settings import latest_export


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _dataset(args: argparse.Namespace) -> Path:
    """Explicit --dataset, otherwise the newest export in $FIGURE_PROJECT_DIR/exports."""
    return args.dataset if args.dataset is not None else latest_export()


def _cache_root(args: argparse.Namespace) -> Path | None:
    return None if args.no_cache else DEFAULT_CACHE_ROOT


def cmd_run(args: argparse.Namespace) -> int:
    result = build_pipeline(args.pipeline).run(args.input, args.output)
    print(f"pipeline={args.pipeline}")
    print(f"output={result.output}")
    print(f"quality_score={result.quality.score:.4f}")
    print(f"requires_review={result.quality.requires_review}")
    print(f"metadata={json.dumps(result.metadata, ensure_ascii=False)}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    report = benchmark_pipeline(
        args.pipeline,
        _dataset(args),
        split=args.split,
        write_debug=not args.no_debug,
        cache_root=_cache_root(args),
    )
    _print_json(report)
    return 1 if report["failure_count"] else 0


def _summarize(report: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "pipeline_id": report["pipeline_id"],
        "run_id": report["run_id"],
        "success_count": report["success_count"],
        "failure_count": report["failure_count"],
        "device": report["device"],
        "latency_ms_p50": report["latency_ms"]["p50"],
        "cache_hits": report["cache"]["hits"],
        "cache_misses": report["cache"]["misses"],
        "benchmark_path": report["benchmark_path"],
    }
    if report.get("skipped"):
        summary.update(skipped=True, error=report["error"])
    return summary


def cmd_eval(args: argparse.Namespace) -> int:
    dataset = _dataset(args)
    reports = eval_pipelines(
        dataset,
        pipeline_ids=args.pipelines,
        split=args.split,
        write_debug=not args.no_debug,
        cache_root=_cache_root(args),
    )
    _print_json({"dataset": str(dataset), "runs": [_summarize(report) for report in reports]})
    failed = any(report["failure_count"] for report in reports)
    # Skipped pipelines only fail the run when explicitly requested.
    skipped_requested = bool(args.pipelines) and any(report.get("skipped") for report in reports)
    return 1 if failed or skipped_requested else 0


def cmd_prepare_dataset(args: argparse.Namespace) -> int:
    root = prepare_eval_set(args.output, count=args.count, seed=args.seed, overwrite=args.force)
    print(f"dataset={root}")
    print(f"images={args.count}")
    return 0


def cmd_init_dataset(args: argparse.Namespace) -> int:
    summary = init_export_dataset(
        _dataset(args),
        categories=args.categories or ["FIGURE"],
        roles=args.roles or ["main", "detail"],
        min_side=args.min_side,
        max_aspect=args.max_aspect,
    )
    _print_json({key: value for key, value in summary.items() if key != "rejected"})
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    summary = compare_runs(_dataset(args), args.pipelines, split=args.split)
    _print_json(
        {
            "output_dir": summary["output_dir"],
            "review_csv": summary["review_csv"],
            "pipelines": summary["pipelines"],
        }
    )
    return 0


def cmd_list_pipelines(_: argparse.Namespace) -> int:
    print("\n".join(list_pipelines()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="figure-cutout",
        description="Figure-specific precision cutout CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_dataset_option(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--dataset",
            type=Path,
            default=None,
            help="Dataset directory. Default: newest export in $FIGURE_PROJECT_DIR/exports "
            "(~/figure_project/exports).",
        )

    def add_bench_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--split", default="val")
        p.add_argument("--no-debug", action="store_true", help="Skip debug artifacts.")
        p.add_argument(
            "--no-cache",
            action="store_true",
            help="Re-run inference for every image; use for latency benchmarks.",
        )

    run_parser = sub.add_parser("run", help="Cut out a single image.")
    run_parser.add_argument("input", type=Path)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    run_parser.set_defaults(func=cmd_run)

    bench_parser = sub.add_parser("bench", help="Benchmark one pipeline on a dataset split.")
    add_dataset_option(bench_parser)
    bench_parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    add_bench_options(bench_parser)
    bench_parser.set_defaults(func=cmd_bench)

    eval_parser = sub.add_parser(
        "eval",
        help="Benchmark registered pipelines on one split; write benchmark JSON + debug artifacts.",
    )
    add_dataset_option(eval_parser)
    eval_parser.add_argument(
        "--pipeline",
        action="append",
        dest="pipelines",
        help="Pipeline id (repeatable). Default: all registered pipelines.",
    )
    add_bench_options(eval_parser)
    eval_parser.set_defaults(func=cmd_eval)

    prepare_parser = sub.add_parser(
        "prepare-dataset",
        help="Scaffold an evaluation set with synthetic images and metadata.",
    )
    prepare_parser.add_argument("--output", type=Path, default=Path("datasets/figure-v1"))
    prepare_parser.add_argument("--count", type=int, default=50)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--force", action="store_true", help="Overwrite existing dataset.")
    prepare_parser.set_defaults(func=cmd_prepare_dataset)

    init_parser = sub.add_parser(
        "init-dataset",
        help="Index an image export (index.json) in place: metadata stubs + val split.",
    )
    add_dataset_option(init_parser)
    init_parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="Export `category` value to keep (repeatable). Default: FIGURE.",
    )
    init_parser.add_argument(
        "--role",
        action="append",
        dest="roles",
        choices=["main", "detail"],
        help="Image role to keep (repeatable). Default: main and detail.",
    )
    init_parser.add_argument("--min-side", type=int, default=256)
    init_parser.add_argument(
        "--max-aspect", type=float, default=3.0, help="Reject long banners beyond this ratio."
    )
    init_parser.set_defaults(func=cmd_init_dataset)

    compare_parser = sub.add_parser(
        "compare",
        help="Side-by-side sheets + review.csv from the latest run of each pipeline.",
    )
    add_dataset_option(compare_parser)
    compare_parser.add_argument(
        "--pipeline",
        action="append",
        dest="pipelines",
        required=True,
        help="Pipeline id (repeatable).",
    )
    compare_parser.add_argument("--split", default="val")
    compare_parser.set_defaults(func=cmd_compare)

    list_parser = sub.add_parser("list-pipelines", help="List registered pipeline ids.")
    list_parser.set_defaults(func=cmd_list_pipelines)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # FIGURE_PROJECT_DIR; existing environment variables win
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
