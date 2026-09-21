from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Detection:
    bbox: tuple[int, int, int, int]
    score: float
    label: str = "figure"


@dataclass(slots=True)
class SegmentationResult:
    mask: Any
    confidence: float


@dataclass(slots=True)
class QualityResult:
    score: float
    requires_review: bool
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CutoutOptions:
    base_policy: str = "include"
    accessory_policy: str = "include_attached"


@dataclass(slots=True)
class CutoutResult:
    source: Path
    output: Path
    quality: QualityResult
    metadata: dict[str, Any] = field(default_factory=dict)
