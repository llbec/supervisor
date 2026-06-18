from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.inference.base import Detection, FrameContext
from app.labels import HOT_WORK_LABELS, PERSON_LABELS, SMOKING_LABELS


@dataclass(frozen=True)
class PersonFrame:
    timestamp_ms: int
    frame_index: int
    bbox: tuple[float, float, float, float]
    keypoints: list[tuple[float, float, float]] = field(default_factory=list)


@dataclass(frozen=True)
class ActivityContext:
    smoking_candidate: bool
    smoking_confidence: float
    hot_work_candidate: bool
    hot_work_confidence: float
    pose_summary: dict[str, Any]


class TrajectoryBuffer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.people: dict[int, deque[PersonFrame]] = {}
        self._next_id = 1
        self.last_analysis_ms = -settings.activity_analysis_interval_ms

    def update(
        self, context: FrameContext, detections: list[Detection]
    ) -> ActivityContext:
        person_detections = [
            detection
            for detection in detections
            if detection.label in PERSON_LABELS and detection.bbox is not None
        ]
        for detection in person_detections:
            track_id = detection.track_id or self._match_or_create_track(detection)
            frames = self.people.setdefault(track_id, deque())
            frames.append(
                PersonFrame(
                    timestamp_ms=context.timestamp_ms,
                    frame_index=context.frame_index,
                    bbox=detection.bbox,
                    keypoints=detection.metadata.get("keypoints", []),
                )
            )

        self._trim(context.timestamp_ms)
        return self._build_activity_context(detections)

    def should_analyze(self, timestamp_ms: int) -> bool:
        if timestamp_ms - self.last_analysis_ms < self.settings.activity_analysis_interval_ms:
            return False
        self.last_analysis_ms = timestamp_ms
        return bool(self.people)

    def summary(self) -> dict[str, Any]:
        people = []
        for track_id, frames in self.people.items():
            if not frames:
                continue
            first = frames[0]
            last = frames[-1]
            people.append(
                {
                    "track_id": track_id,
                    "frames": len(frames),
                    "start_ms": first.timestamp_ms,
                    "end_ms": last.timestamp_ms,
                    "start_bbox": first.bbox,
                    "end_bbox": last.bbox,
                    "vertical_delta": last.bbox[1] - first.bbox[1],
                    "has_pose": any(frame.keypoints for frame in frames),
                    "pose_samples": [
                        {
                            "timestamp_ms": frame.timestamp_ms,
                            "bbox": frame.bbox,
                            "keypoints": frame.keypoints[:17],
                        }
                        for frame in list(frames)[-5:]
                    ],
                }
            )
        return {"people": people}

    def _build_activity_context(self, detections: list[Detection]) -> ActivityContext:
        labels = {d.label for d in detections}
        smoking_confidence = _max_confidence(detections, SMOKING_LABELS)
        hot_work_confidence = _max_confidence(detections, HOT_WORK_LABELS)
        pose_summary = self.summary()
        hand_near_face = _any_hand_near_face(pose_summary)
        work_pose = _any_work_pose(pose_summary)
        return ActivityContext(
            smoking_candidate=bool(labels & SMOKING_LABELS) and hand_near_face,
            smoking_confidence=smoking_confidence if hand_near_face else smoking_confidence * 0.5,
            hot_work_candidate=bool(labels & HOT_WORK_LABELS) and work_pose,
            hot_work_confidence=hot_work_confidence if work_pose else hot_work_confidence * 0.5,
            pose_summary=pose_summary,
        )

    def _trim(self, now_ms: int) -> None:
        min_ms = now_ms - self.settings.trajectory_window_ms
        empty_ids = []
        for track_id, frames in self.people.items():
            while frames and frames[0].timestamp_ms < min_ms:
                frames.popleft()
            if not frames:
                empty_ids.append(track_id)
        for track_id in empty_ids:
            self.people.pop(track_id, None)

    def _match_or_create_track(self, detection: Detection) -> int:
        assert detection.bbox is not None
        best_id = None
        best_iou = 0.0
        for track_id, frames in self.people.items():
            if not frames:
                continue
            value = _iou(detection.bbox, frames[-1].bbox)
            if value > best_iou:
                best_iou = value
                best_id = track_id
        if best_id is not None and best_iou >= self.settings.tracker_iou_threshold:
            return best_id
        track_id = self._next_id
        self._next_id += 1
        return track_id


def _max_confidence(detections: list[Detection], labels: set[str]) -> float:
    values = [d.confidence for d in detections if d.label in labels]
    return max(values) if values else 0.0


def _any_hand_near_face(summary: dict[str, Any]) -> bool:
    for person in summary.get("people", []):
        for sample in person.get("pose_samples", []):
            keypoints = sample.get("keypoints") or []
            if len(keypoints) < 11:
                continue
            nose = keypoints[0]
            wrists = [keypoints[9], keypoints[10]]
            face_scale = max(20.0, _bbox_height(sample["bbox"]) * 0.18)
            if any(_distance(nose, wrist) < face_scale for wrist in wrists):
                return True
    return False


def _any_work_pose(summary: dict[str, Any]) -> bool:
    for person in summary.get("people", []):
        for sample in person.get("pose_samples", []):
            keypoints = sample.get("keypoints") or []
            if len(keypoints) < 11:
                continue
            shoulders = [keypoints[5], keypoints[6]]
            wrists = [keypoints[9], keypoints[10]]
            if any(wrist[1] < shoulder[1] for wrist in wrists for shoulder in shoulders):
                return True
    return False


def _bbox_height(bbox: tuple[float, float, float, float]) -> float:
    return max(1.0, bbox[3] - bbox[1])


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    if len(a) < 3 or len(b) < 3 or a[2] < 0.2 or b[2] < 0.2:
        return 999999.0
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


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
