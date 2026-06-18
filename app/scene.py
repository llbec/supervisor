from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.inference.base import Detection, FrameContext


IGNORED_SCENE_LABELS = {
    "person",
    "worker",
    "helmet",
    "hardhat",
    "safety_helmet",
    "vest",
    "safety_vest",
    "reflective_vest",
    "hi_vis_vest",
}


@dataclass(frozen=True)
class SceneCandidate:
    signature: tuple[str, ...]
    reason: str


class SceneSampler:
    """Pick one representative frame whenever YOLO context changes."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.current_signature: tuple[str, ...] | None = None
        self.last_emit_ms = -settings.scene_min_change_interval_ms
        self.seen_signatures: set[tuple[str, ...]] = set()

    def update(
        self, context: FrameContext, detections: list[Detection]
    ) -> SceneCandidate | None:
        signature = _scene_signature(detections)
        if not signature:
            signature = ("unknown",)
        if signature in self.seen_signatures:
            self.current_signature = signature
            return None
        if (
            self.current_signature == signature
            or context.timestamp_ms - self.last_emit_ms
            < self.settings.scene_min_change_interval_ms
        ):
            return None

        reason = "initial_scene" if self.current_signature is None else "scene_changed"
        self.current_signature = signature
        self.seen_signatures.add(signature)
        self.last_emit_ms = context.timestamp_ms
        return SceneCandidate(signature=signature, reason=reason)


def _scene_signature(detections: list[Detection]) -> tuple[str, ...]:
    labels = sorted(
        {
            detection.label
            for detection in detections
            if detection.label not in IGNORED_SCENE_LABELS
            and detection.confidence >= 0.35
        }
    )
    return tuple(labels[:12])
