from __future__ import annotations

from typing import Any, Protocol

from figure_cutout.domain.models import Detection, QualityResult, SegmentationResult


class Detector(Protocol):
    name: str

    def detect(self, image: Any) -> list[Detection]:
        ...


class Segmenter(Protocol):
    name: str

    def segment(self, image: Any, detection: Detection) -> SegmentationResult:
        ...


class MaskRefiner(Protocol):
    name: str

    def refine(self, image: Any, mask: Any) -> Any:
        ...


class QualityEvaluator(Protocol):
    name: str

    def evaluate(
        self,
        image: Any,
        detection: Detection,
        mask: Any,
        segmentation_confidence: float,
    ) -> QualityResult:
        ...
