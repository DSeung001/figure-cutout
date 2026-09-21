from figure_cutout.ml.pipeline import FigureCutoutPipeline
from figure_cutout.ml.placeholder import (
    FullFrameDetector,
    IdentityRefiner,
    OpaqueSegmenter,
    PlaceholderQualityEvaluator,
)


def build_local_placeholder_pipeline() -> FigureCutoutPipeline:
    """Builds an end-to-end pipeline for validating orchestration before real models land."""
    return FigureCutoutPipeline(
        detector=FullFrameDetector(),
        segmenter=OpaqueSegmenter(),
        refiner=IdentityRefiner(),
        quality_evaluator=PlaceholderQualityEvaluator(),
    )
