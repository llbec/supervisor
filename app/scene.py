from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from app.inference.base import Detection, FrameContext
from app.labels import HELMET_LABELS, PERSON_LABELS, VEST_LABELS


IGNORED_SCENE_LABELS = PERSON_LABELS | HELMET_LABELS | VEST_LABELS


@dataclass
class RepresentativeFrame:
    frame: Any
    context: FrameContext
    detections: list[Detection]
    score: int


class SceneAggregator:
    def __init__(self):
        self.label_counts: Counter[str] = Counter()
        self.representative: RepresentativeFrame | None = None

    def update(self, frame, context: FrameContext, detections: list[Detection]) -> None:
        labels = [d.label for d in detections if d.label not in IGNORED_SCENE_LABELS]
        self.label_counts.update(labels)
        score = len(set(labels)) * 5 + len(detections)
        if self.representative is None or score > self.representative.score:
            self.representative = RepresentativeFrame(
                frame=frame.copy(),
                context=context,
                detections=list(detections),
                score=score,
            )

    def summary(self) -> dict:
        return {
            "scene_labels": self.label_counts.most_common(20),
        }
