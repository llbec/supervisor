from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.inference.base import Detection, FrameContext
from app.labels import (
    DOCUMENT_LABELS,
    HEIGHT_CONTEXT_LABELS,
    HOT_WORK_LABELS,
    PERSON_LABELS,
    SMOKING_LABELS,
)


@dataclass
class CandidateSegment:
    candidate_id: str
    task: str
    start_ms: int
    end_ms: int
    key_frame_ms: int
    evidence: dict[str, Any]
    snapshots: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "task": self.task,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.end_ms - self.start_ms,
            "key_frame_ms": self.key_frame_ms,
            "evidence": self.evidence,
            "snapshots": self.snapshots,
        }


class ActivityAnalyzer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.frames: deque[dict[str, Any]] = deque()
        self.label_counts: Counter[str] = Counter()
        self.height_candidates: list[CandidateSegment] = []
        self.briefing_candidates: list[CandidateSegment] = []
        self._height_open: dict[str, Any] | None = None
        self._briefing_open: dict[str, Any] | None = None

    def update(self, frame: FrameContext, detections: list[Detection]) -> dict[str, Any]:
        labels = {d.label for d in detections}
        for label in labels:
            self.label_counts[label] += 1
        pose_signals = _pose_signals(detections)
        state = {
            "timestamp_ms": frame.timestamp_ms,
            "frame_index": frame.frame_index,
            "labels": sorted(labels),
            "person_count": sum(1 for d in detections if d.label in PERSON_LABELS),
            "pose_signals": pose_signals,
        }
        self.frames.append(state)
        self._trim(frame.timestamp_ms)
        self._update_height_candidate(state)
        self._update_briefing_candidate(state)
        return {
            "smoking_candidate": bool(labels & SMOKING_LABELS)
            and "hand_near_face" in pose_signals,
            "smoking_confidence": _max_confidence(detections, SMOKING_LABELS),
            "hot_work_candidate": bool(labels & HOT_WORK_LABELS)
            and ("work_pose" in pose_signals or "arms_raised" in pose_signals),
            "hot_work_confidence": _max_confidence(detections, HOT_WORK_LABELS),
            "pose_signals": pose_signals,
        }

    def finalize(self) -> None:
        if self._height_open:
            self.height_candidates.append(_close_segment(self._height_open))
            self._height_open = None
        if self._briefing_open:
            self.briefing_candidates.append(_close_segment(self._briefing_open))
            self._briefing_open = None

    def summary(self) -> dict[str, Any]:
        return {
            "top_labels": self.label_counts.most_common(30),
            "recent_frames": list(self.frames)[-20:],
        }

    def _trim(self, now_ms: int) -> None:
        min_ms = now_ms - self.settings.trajectory_window_ms
        while self.frames and self.frames[0]["timestamp_ms"] < min_ms:
            self.frames.popleft()

    def _update_height_candidate(self, state: dict[str, Any]) -> None:
        labels = set(state["labels"])
        active = bool(labels & HEIGHT_CONTEXT_LABELS) and bool(
            {"climb_pose", "arms_raised", "work_pose"} & set(state["pose_signals"])
        )
        self._height_open = _update_open_segment(
            self._height_open,
            active,
            state,
            "height_work",
            self.height_candidates,
            self.settings.min_candidate_duration_ms,
            {"objects": sorted(labels & HEIGHT_CONTEXT_LABELS)},
        )

    def _update_briefing_candidate(self, state: dict[str, Any]) -> None:
        labels = set(state["labels"])
        document_visible = bool(labels & DOCUMENT_LABELS)
        active = state["person_count"] >= 2 and (
            document_visible
            or "explain_gesture" in state["pose_signals"]
            or "hand_near_document" in state["pose_signals"]
        )
        self._briefing_open = _update_open_segment(
            self._briefing_open,
            active,
            state,
            "briefing",
            self.briefing_candidates,
            self.settings.min_candidate_duration_ms,
            {"document_labels": sorted(labels & DOCUMENT_LABELS)},
        )


def _update_open_segment(
    current: dict[str, Any] | None,
    active: bool,
    state: dict[str, Any],
    task: str,
    target: list[CandidateSegment],
    min_duration_ms: int,
    evidence: dict[str, Any],
) -> dict[str, Any] | None:
    if active and current is None:
        return {
            "task": task,
            "start_ms": state["timestamp_ms"],
            "end_ms": state["timestamp_ms"],
            "key_frame_ms": state["timestamp_ms"],
            "evidence": {"pose_signals": [], **evidence},
        }
    if active and current is not None:
        current["end_ms"] = state["timestamp_ms"]
        current["key_frame_ms"] = state["timestamp_ms"]
        current["evidence"]["pose_signals"] = sorted(
            set(current["evidence"].get("pose_signals", [])) | set(state["pose_signals"])
        )
        return current
    if not active and current is not None:
        segment = _close_segment(current)
        if segment.end_ms - segment.start_ms >= min_duration_ms:
            target.append(segment)
    return None


def _close_segment(data: dict[str, Any]) -> CandidateSegment:
    return CandidateSegment(
        candidate_id=f"{data['task']}_{data['start_ms']}_{data['end_ms']}",
        task=data["task"],
        start_ms=data["start_ms"],
        end_ms=data["end_ms"],
        key_frame_ms=data["key_frame_ms"],
        evidence=data["evidence"],
    )


def _pose_signals(detections: list[Detection]) -> list[str]:
    signals: set[str] = set()
    for detection in detections:
        keypoints = detection.metadata.get("keypoints") or []
        if len(keypoints) < 11:
            continue
        nose = keypoints[0]
        shoulders = [keypoints[5], keypoints[6]]
        wrists = [keypoints[9], keypoints[10]]
        bbox = detection.bbox
        if bbox is None:
            continue
        scale = max(20.0, (bbox[3] - bbox[1]) * 0.18)
        if any(_distance(nose, wrist) < scale for wrist in wrists):
            signals.add("hand_near_face")
        if any(wrist[1] < shoulder[1] for wrist in wrists for shoulder in shoulders):
            signals.add("arms_raised")
            signals.add("work_pose")
        if any(wrist[2] > 0.2 for wrist in wrists):
            signals.add("explain_gesture")
    return sorted(signals)


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    if len(a) < 3 or len(b) < 3 or a[2] < 0.2 or b[2] < 0.2:
        return 999999.0
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _max_confidence(detections: list[Detection], labels: set[str]) -> float:
    values = [d.confidence for d in detections if d.label in labels]
    return max(values) if values else 0.0
