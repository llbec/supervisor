from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.inference.base import Detection
from app.rules import HELMET_LABELS, VEST_LABELS

if TYPE_CHECKING:
    from app.config import Settings


PERSON_LABELS = {"person", "worker"}


@dataclass
class PersonPPEState:
    track_id: int
    bbox: tuple[float, float, float, float]
    seen_frames: int = 0
    helmet_hits: int = 0
    vest_hits: int = 0
    missing_frames: int = 0
    last_frame_index: int = 0

    @property
    def helmet_ok(self) -> bool:
        return self.helmet_hits > 0

    @property
    def vest_ok(self) -> bool:
        return self.vest_hits > 0


@dataclass(frozen=True)
class PPESummary:
    person_count: int
    helmet_count: int
    vest_count: int
    missing_helmet: bool
    missing_vest: bool
    tracked_people: list[dict] = field(default_factory=list)


class PPETracker:
    """Track people across frames and associate nearby PPE detections."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.states: dict[int, PersonPPEState] = {}
        self._next_id = 1

    def update(self, detections: list[Detection], frame_index: int) -> PPESummary:
        person_detections = [
            detection
            for detection in detections
            if detection.label in PERSON_LABELS and detection.bbox is not None
        ]
        helmet_boxes = [
            detection.bbox
            for detection in detections
            if detection.label in HELMET_LABELS and detection.bbox is not None
        ]
        vest_boxes = [
            detection.bbox
            for detection in detections
            if detection.label in VEST_LABELS and detection.bbox is not None
        ]

        active_ids: set[int] = set()
        for person in person_detections:
            track_id = person.track_id or self._match_or_create_track(person.bbox)
            active_ids.add(track_id)
            state = self.states.get(track_id)
            if state is None:
                state = PersonPPEState(track_id=track_id, bbox=person.bbox)
                self.states[track_id] = state
            state.bbox = person.bbox
            state.seen_frames += 1
            state.last_frame_index = frame_index

            if _has_helmet(person.bbox, helmet_boxes):
                state.helmet_hits += 1
            if _has_vest(person.bbox, vest_boxes):
                state.vest_hits += 1

            if not state.helmet_ok or not state.vest_ok:
                state.missing_frames += 1
            else:
                state.missing_frames = 0

        self._drop_stale_tracks(frame_index)

        visible_states = [
            state for track_id, state in self.states.items() if track_id in active_ids
        ]
        confirmed_states = [
            state
            for state in visible_states
            if state.seen_frames >= self.settings.ppe_required_hits
        ]
        people_for_decision = confirmed_states or visible_states
        missing_helmet = any(
            not state.helmet_ok
            and state.missing_frames > self.settings.ppe_missing_tolerance
            for state in people_for_decision
        )
        missing_vest = any(
            not state.vest_ok
            and state.missing_frames > self.settings.ppe_missing_tolerance
            for state in people_for_decision
        )

        return PPESummary(
            person_count=len(visible_states),
            helmet_count=sum(1 for state in visible_states if state.helmet_ok),
            vest_count=sum(1 for state in visible_states if state.vest_ok),
            missing_helmet=missing_helmet,
            missing_vest=missing_vest,
            tracked_people=[
                {
                    "track_id": state.track_id,
                    "seen_frames": state.seen_frames,
                    "helmet_hits": state.helmet_hits,
                    "vest_hits": state.vest_hits,
                    "missing_frames": state.missing_frames,
                }
                for state in visible_states
            ],
        )

    def _match_or_create_track(self, bbox: tuple[float, float, float, float]) -> int:
        best_id = None
        best_iou = 0.0
        for track_id, state in self.states.items():
            value = _iou(bbox, state.bbox)
            if value > best_iou:
                best_iou = value
                best_id = track_id
        if best_id is not None and best_iou >= self.settings.tracker_iou_threshold:
            return best_id
        track_id = self._next_id
        self._next_id += 1
        return track_id

    def _drop_stale_tracks(self, frame_index: int) -> None:
        max_age = max(self.settings.frame_sample_interval * 4, 20)
        stale_ids = [
            track_id
            for track_id, state in self.states.items()
            if frame_index - state.last_frame_index > max_age
        ]
        for track_id in stale_ids:
            self.states.pop(track_id, None)


def _has_helmet(
    person_box: tuple[float, float, float, float],
    helmet_boxes: list[tuple[float, float, float, float]],
) -> bool:
    px1, py1, px2, py2 = person_box
    head_region = (px1, py1, px2, py1 + (py2 - py1) * 0.35)
    return any(_intersection_over_box(box, head_region) > 0.15 for box in helmet_boxes)


def _has_vest(
    person_box: tuple[float, float, float, float],
    vest_boxes: list[tuple[float, float, float, float]],
) -> bool:
    px1, py1, px2, py2 = person_box
    torso_region = (
        px1,
        py1 + (py2 - py1) * 0.25,
        px2,
        py1 + (py2 - py1) * 0.8,
    )
    return any(_intersection_over_box(box, torso_region) > 0.2 for box in vest_boxes)


def _intersection_over_box(
    box: tuple[float, float, float, float],
    region: tuple[float, float, float, float],
) -> float:
    x1 = max(box[0], region[0])
    y1 = max(box[1], region[1])
    x2 = min(box[2], region[2])
    y2 = min(box[3], region[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    return intersection / box_area


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0
