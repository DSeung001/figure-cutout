from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from figure_cutout.domain.models import CutoutOptions, CutoutResult
from figure_cutout.ml.contracts import Detector, MaskRefiner, QualityEvaluator, Segmenter


class FigureCutoutPipeline:
    version = "0.1.0"

    def __init__(
        self,
        detector: Detector,
        segmenter: Segmenter,
        refiner: MaskRefiner,
        quality_evaluator: QualityEvaluator,
    ) -> None:
        self.detector = detector
        self.segmenter = segmenter
        self.refiner = refiner
        self.quality_evaluator = quality_evaluator

    def run(
        self,
        source: Path,
        output: Path,
        options: CutoutOptions | None = None,
    ) -> CutoutResult:
        options = options or CutoutOptions()

        rgba = Image.open(source).convert("RGBA")
        image = np.asarray(rgba)

        detections = self.detector.detect(image)
        if not detections:
            raise RuntimeError("No figure candidate detected.")

        target = max(detections, key=lambda item: item.score)
        segmentation = self.segmenter.segment(image, target)
        mask = self.refiner.refine(image, segmentation.mask)

        quality = self.quality_evaluator.evaluate(
            image=image,
            detection=target,
            mask=mask,
            segmentation_confidence=segmentation.confidence,
        )

        alpha = Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L")
        result = rgba.copy()
        result.putalpha(alpha)

        output.parent.mkdir(parents=True, exist_ok=True)
        result.save(output)

        return CutoutResult(
            source=source,
            output=output,
            quality=quality,
            metadata={
                "pipeline_version": self.version,
                "detector": self.detector.name,
                "segmenter": self.segmenter.name,
                "refiner": self.refiner.name,
                "quality_evaluator": self.quality_evaluator.name,
                "options": asdict(options),
                "detection_score": target.score,
                "segmentation_confidence": segmentation.confidence,
            },
        )
