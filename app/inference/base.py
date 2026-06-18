from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float] | None = None
    track_id: int | None = None
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameContext:
    frame_index: int
    timestamp_ms: int
    width: int
    height: int


@dataclass(frozen=True)
class PoseObservation:
    track_id: int | None
    bbox: tuple[float, float, float, float] | None
    keypoints: list[tuple[float, float, float]]
    confidence: float
