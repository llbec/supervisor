from __future__ import annotations

from typing import Any

from app.config import Settings
from app.inference.base import Detection


class VisionLanguageVerifier:
    """Verifier hook for Qwen-VL.

    The default implementation is intentionally conservative. It provides a
    stable interface for the processor and can be replaced with a GPU-backed
    Qwen implementation without changing API or database code.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def classify_scene(self, frame: Any, detections: list[Detection]) -> tuple[str, float]:
        labels = {d.label for d in detections}
        if labels & {"server_rack", "cabinet", "machine_room", "equipment_room"}:
            return "machine_room", 0.75
        if labels & {"tower", "telecom_tower", "pylon"}:
            return "near_tower", 0.75
        return "other", 0.4

    def confirm_briefing(self, frame: Any, detections: list[Detection]) -> tuple[bool, float]:
        person_count = sum(1 for d in detections if d.label == "person")
        if person_count >= 3:
            return True, 0.45
        return False, 0.2

    def confirm_height_work(self, frame: Any, detections: list[Detection]) -> tuple[bool, float]:
        labels = {d.label for d in detections}
        if labels & {"ladder", "scaffold", "tower", "telecom_tower", "aerial_lift"}:
            return True, 0.7
        return False, 0.25
