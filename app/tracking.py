from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.inference.base import Detection
from app.labels import HELMET_LABELS, PERSON_LABELS, VEST_LABELS

if TYPE_CHECKING:
    from app.config import Settings


@dataclass
class PersonPPEState:
    track_id: int
    bbox: tuple[float, float, float, float]
    seen_frames: int = 0
    helmet_hits: int = 0
    vest_hits: int = 0
    missing_frames: int = 0
    last_frame_index: int = 0
    helmet_exempt: bool = False
    vest_exempt: bool = False
    exemption_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PPESummary:
    person_count: int
    helmet_count: int
    vest_count: int
    missing_helmet: bool
    missing_vest: bool
    tracked_people: list[dict]
    exempt_people: list[dict]


class PPETracker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.states: dict[int, PersonPPEState] = {}
        self._next_id = 1

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        frame_width: int,
        frame_height: int,
    ) -> PPESummary:
        people = [d for d in detections if d.label in PERSON_LABELS and d.bbox]
        helmets = [d.bbox for d in detections if d.label in HELMET_LABELS and d.bbox]
        vests = [d.bbox for d in detections if d.label in VEST_LABELS and d.bbox]
        active_ids: set[int] = set()

        for person in people:
            assert person.bbox is not None
            track_id = person.track_id or self._match_or_create_track(person.bbox)
            active_ids.add(track_id)
            state = self.states.get(track_id) or PersonPPEState(track_id, person.bbox)
            state.bbox = person.bbox
            state.seen_frames += 1
            state.last_frame_index = frame_index
            reasons = _ppe_exemptions(
                person.bbox,
                frame_width,
                frame_height,
                self.settings.ppe_edge_margin_ratio,
            )
            state.helmet_exempt = "helmet" in reasons
            state.vest_exempt = "vest" in reasons
            state.exemption_reasons = reasons
            if _has_helmet(person.bbox, helmets):
                state.helmet_hits += 1
            if _has_vest(person.bbox, vests):
                state.vest_hits += 1
            if (state.helmet_hits == 0 and not state.helmet_exempt) or (
                state.vest_hits == 0 and not state.vest_exempt
            ):
                state.missing_frames += 1
            else:
                state.missing_frames = 0
            self.states[track_id] = state

        visible = [s for tid, s in self.states.items() if tid in active_ids]
        confirmed = [s for s in visible if s.seen_frames >= self.settings.ppe_required_hits]
        decision = confirmed or visible
        missing_helmet = any(
            s.helmet_hits == 0
            and not s.helmet_exempt
            and s.missing_frames > self.settings.ppe_missing_tolerance
            for s in decision
        )
        missing_vest = any(
            s.vest_hits == 0
            and not s.vest_exempt
            and s.missing_frames > self.settings.ppe_missing_tolerance
            for s in decision
        )
        return PPESummary(
            person_count=len(visible),
            helmet_count=sum(1 for s in visible if s.helmet_hits > 0),
            vest_count=sum(1 for s in visible if s.vest_hits > 0),
            missing_helmet=missing_helmet,
            missing_vest=missing_vest,
            tracked_people=[_state_details(s) for s in visible],
            exempt_people=[_state_details(s) for s in visible if s.helmet_exempt or s.vest_exempt],
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


def _state_details(state: PersonPPEState) -> dict:
    return {
        "track_id": state.track_id,
        "seen_frames": state.seen_frames,
        "helmet_hits": state.helmet_hits,
        "vest_hits": state.vest_hits,
        "missing_frames": state.missing_frames,
        "helmet_exempt": state.helmet_exempt,
        "vest_exempt": state.vest_exempt,
        "exemption_reasons": state.exemption_reasons,
    }


def _has_helmet(person: tuple[float, float, float, float], helmets: list[tuple[float, float, float, float]]) -> bool:
    x1, y1, x2, y2 = person
    head = (x1, y1, x2, y1 + (y2 - y1) * 0.35)
    return any(_intersection_over_box(box, head) > 0.15 for box in helmets)


def _has_vest(person: tuple[float, float, float, float], vests: list[tuple[float, float, float, float]]) -> bool:
    x1, y1, x2, y2 = person
    torso = (x1, y1 + (y2 - y1) * 0.25, x2, y1 + (y2 - y1) * 0.8)
    return any(_intersection_over_box(box, torso) > 0.2 for box in vests)


def _ppe_exemptions(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    margin_ratio: float,
) -> list[str]:
    x1, y1, x2, y2 = box
    mx = width * margin_ratio
    my = height * margin_ratio
    reasons: list[str] = []
    if y1 <= my:
        reasons += ["helmet", "head_out_of_frame"]
    if x1 <= mx or x2 >= width - mx:
        reasons += ["helmet", "vest", "body_out_of_frame"]
    if y2 >= height - my:
        reasons += ["vest", "body_out_of_frame"]
    if (x2 - x1) < width * 0.025 or (y2 - y1) < height * 0.08:
        reasons += ["helmet", "vest", "person_too_small"]
    return sorted(set(reasons))


def _intersection_over_box(
    box: tuple[float, float, float, float],
    region: tuple[float, float, float, float],
) -> float:
    x1 = max(box[0], region[0])
    y1 = max(box[1], region[1])
    x2 = min(box[2], region[2])
    y2 = min(box[3], region[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    return intersection / area


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0
