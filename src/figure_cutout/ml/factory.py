from __future__ import annotations

from collections.abc import Callable

from figure_cutout.ml.pipeline import FigureCutoutPipeline
from figure_cutout.ml.placeholder import (
    FullFrameDetector,
    IdentityRefiner,
    OpaqueSegmenter,
    PlaceholderQualityEvaluator,
)

PipelineBuilder = Callable[[], FigureCutoutPipeline]

DEFAULT_PIPELINE = "placeholder"
PIPELINE_REGISTRY: dict[str, PipelineBuilder] = {}


def register_pipeline(name: str, builder: PipelineBuilder) -> None:
    PIPELINE_REGISTRY[name] = builder


def list_pipelines() -> list[str]:
    return sorted(PIPELINE_REGISTRY)


def build_pipeline(name: str) -> FigureCutoutPipeline:
    try:
        builder = PIPELINE_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(list_pipelines()) or "(none)"
        raise ValueError(f"Unknown pipeline '{name}'. Known: {known}") from exc
    return builder()


def build_local_placeholder_pipeline() -> FigureCutoutPipeline:
    """Builds an end-to-end pipeline for validating orchestration before real models land."""
    return FigureCutoutPipeline(
        detector=FullFrameDetector(),
        segmenter=OpaqueSegmenter(),
        refiner=IdentityRefiner(),
        quality_evaluator=PlaceholderQualityEvaluator(),
    )


def build_rembg_pipeline(model_name: str = "u2net") -> FigureCutoutPipeline:
    from figure_cutout.ml.rembg_adapter import RembgQualityEvaluator, RembgSegmenter

    return FigureCutoutPipeline(
        detector=FullFrameDetector(),
        segmenter=RembgSegmenter(model_name=model_name),
        refiner=IdentityRefiner(),
        quality_evaluator=RembgQualityEvaluator(),
    )


register_pipeline("placeholder", build_local_placeholder_pipeline)
register_pipeline("rembg", build_rembg_pipeline)
