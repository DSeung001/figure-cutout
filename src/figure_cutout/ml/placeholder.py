from __future__ import annotations

from typing import Any

import numpy as np

from figure_cutout.domain.models import Detection, QualityResult, SegmentationResult


class FullFrameDetector:
    """Development placeholder. Replace with a figure-specific detector."""

    name = "placeholder-full-frame"

    def detect(self, image: Any) -> list[Detection]:
        height, width = image.shape[:2]
        return [Detection(bbox=(0, 0, width, height), score=1.0)]


class OpaqueSegmenter:
    """Development placeholder returning an all-foreground mask."""

    name = "placeholder-opaque"

    def segment(self, image: Any, detection: Detection) -> SegmentationResult:
        height, width = image.shape[:2]
        mask = np.full((height, width), 255, dtype=np.uint8)
        return SegmentationResult(mask=mask, confidence=0.0)


class IdentityRefiner:
    name = "placeholder-identity"

    def refine(self, image: Any, mask: Any) -> Any:
        return mask


class PlaceholderQualityEvaluator:
    name = "placeholder-quality"

    def evaluate(
        self,
        image: Any,
        detection: Detection,
        mask: Any,
        segmentation_confidence: float,
    ) -> QualityResult:
        return QualityResult(
            score=segmentation_confidence,
            requires_review=True,
            reasons=["placeholder_model"],
        )
