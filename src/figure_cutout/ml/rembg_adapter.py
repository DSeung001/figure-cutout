from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from figure_cutout.domain.models import Detection, QualityResult, SegmentationResult

REVIEW_FOREGROUND_RATIO = 0.15


def _load_rembg() -> tuple[Any, Any]:
    try:
        from rembg import new_session, remove
    except ImportError as exc:
        raise ImportError(
            "rembg is required for the rembg pipeline. Install with: uv sync --extra ml"
        ) from exc
    return remove, new_session


class RembgSegmenter:
    """Background-removal baseline behind the Segmenter contract."""

    def __init__(self, model_name: str = "u2net") -> None:
        self._remove, self._new_session = _load_rembg()
        self.model_name = model_name
        self.name = f"rembg-{model_name}"
        self._session: Any | None = None

    def _get_session(self) -> Any:
        # Created once and kept warm across calls.
        if self._session is None:
            self._session = self._new_session(self.model_name)
        return self._session

    def segment(self, image: Any, detection: Detection) -> SegmentationResult:
        rgba = Image.fromarray(np.asarray(image), mode="RGBA")
        cutout = self._remove(rgba, session=self._get_session()).convert("RGBA")
        alpha = np.asarray(cutout.split()[-1], dtype=np.uint8)

        height, width = alpha.shape
        x0, y0, x1, y1 = detection.bbox
        x0, x1 = (max(0, min(width, v)) for v in (x0, x1))
        y0, y1 = (max(0, min(height, v)) for v in (y0, y1))
        if (x0, y0, x1, y1) != (0, 0, width, height):
            cropped = np.zeros_like(alpha)
            cropped[y0:y1, x0:x1] = alpha[y0:y1, x0:x1]
            alpha = cropped

        # Foreground ratio heuristic, not a calibrated model confidence.
        foreground_ratio = float(np.count_nonzero(alpha)) / float(alpha.size)
        return SegmentationResult(mask=alpha, confidence=min(1.0, foreground_ratio * 2.0))


class RembgQualityEvaluator:
    name = "rembg-quality"

    def evaluate(
        self,
        image: Any,
        detection: Detection,
        mask: Any,
        segmentation_confidence: float,
    ) -> QualityResult:
        reasons: list[str] = []
        if not np.any(np.asarray(mask) > 0):
            reasons.append("empty_mask")
        elif segmentation_confidence < REVIEW_FOREGROUND_RATIO:
            reasons.append("low_foreground_ratio")
        return QualityResult(
            score=segmentation_confidence,
            requires_review=bool(reasons),
            reasons=reasons,
        )
