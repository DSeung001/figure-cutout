from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from figure_cutout.domain.models import CutoutOptions, CutoutResult, PipelineTrace
from figure_cutout.image_io import load_rgba
from figure_cutout.ml.contracts import Detector, MaskRefiner, QualityEvaluator, Segmenter


class FigureCutoutPipeline:
    # Bump when pipeline or component behavior changes: result caches are keyed on it.
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

    def fingerprint(self) -> str:
        """Identity of this pipeline configuration; changes when version or any model changes."""
        parts = [
            self.version,
            self.detector.name,
            self.segmenter.name,
            self.refiner.name,
            self.quality_evaluator.name,
        ]
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]

    def run(
        self,
        source: Path,
        output: Path,
        options: CutoutOptions | None = None,
    ) -> CutoutResult:
        options = options or CutoutOptions()

        rgba = load_rgba(source)
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

        # Already-transparent input (cut-out product shots) keeps its transparency.
        alpha = Image.fromarray(
            np.minimum(image[..., 3], np.asarray(mask, dtype=np.uint8)), mode="L"
        )
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
            trace=PipelineTrace(
                detection=target,
                raw_mask=segmentation.mask,
                refined_mask=mask,
            ),
        )
