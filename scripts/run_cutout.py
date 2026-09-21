from __future__ import annotations

import argparse
from pathlib import Path

from figure_cutout.ml.factory import build_local_placeholder_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local figure cutout pipeline.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pipeline = build_local_placeholder_pipeline()
    result = pipeline.run(args.input, args.output)

    print(f"output={result.output}")
    print(f"quality_score={result.quality.score:.4f}")
    print(f"requires_review={result.quality.requires_review}")
    print(f"pipeline={result.metadata}")


if __name__ == "__main__":
    main()
