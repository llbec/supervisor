from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from app.inference.base import Detection, FrameContext
from app.labels import HELMET_LABELS, HOT_WORK_LABELS, SMOKING_LABELS, VEST_LABELS


@dataclass(frozen=True)
class Finding:
    event_type: str
    value: str
    confidence: float
    frame: FrameContext
    details: dict
    alert: bool = False
    severity: str = "warning"
    message: str | None = None


class PPESummaryLike(Protocol):
    person_count: int
    helmet_count: int
    vest_count: int
    missing_helmet: bool
    missing_vest: bool
    tracked_people: list[dict]
    exempt_people: list[dict]


class ActivityContextLike(Protocol):
    smoking_candidate: bool
    smoking_confidence: float
    hot_work_candidate: bool
    hot_work_confidence: float
    pose_summary: dict[str, Any]


def summarize_frame(
    frame: FrameContext,
    detections: list[Detection],
    scene: tuple[str, float],
    briefing: tuple[bool, float],
    height_work: tuple[bool, float],
    ppe_summary: PPESummaryLike | None = None,
    activity_context: ActivityContextLike | None = None,
    scene_snapshot: str | None = None,
    activity_snapshot: str | None = None,
    activity_reason: str | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    labels = {d.label for d in detections}

    if scene[0] != "unknown":
        findings.append(
            Finding(
                "scene",
                scene[0],
                scene[1],
                frame,
                {"labels": sorted(labels), "snapshot_path": scene_snapshot},
            )
        )
    if height_work[1] > 0:
        findings.append(
            Finding(
                "height_work",
                str(height_work[0]).lower(),
                height_work[1],
                frame,
                {
                    "labels": sorted(labels),
                    "snapshot_path": activity_snapshot if height_work[0] else None,
                    "reason": activity_reason,
                    "pose_summary": activity_context.pose_summary
                    if activity_context is not None
                    else {},
                },
            )
        )
    if briefing[1] > 0:
        findings.append(
            Finding(
                "briefing",
                str(briefing[0]).lower(),
                briefing[1],
                frame,
                {
                    "person_count": sum(1 for d in detections if d.label == "person"),
                    "snapshot_path": activity_snapshot if briefing[0] else None,
                    "reason": activity_reason,
                    "pose_summary": activity_context.pose_summary
                    if activity_context is not None
                    else {},
                },
            )
        )

    person_count = (
        ppe_summary.person_count
        if ppe_summary is not None
        else sum(1 for d in detections if d.label == "person")
    )
    helmet_count = (
        ppe_summary.helmet_count
        if ppe_summary is not None
        else sum(1 for d in detections if d.label in HELMET_LABELS)
    )
    vest_count = (
        ppe_summary.vest_count
        if ppe_summary is not None
        else sum(1 for d in detections if d.label in VEST_LABELS)
    )
    if person_count:
        missing_helmet = (
            ppe_summary.missing_helmet
            if ppe_summary is not None
            else helmet_count < person_count
        )
        missing_vest = (
            ppe_summary.missing_vest
            if ppe_summary is not None
            else vest_count < person_count
        )
        ppe_ok = not missing_helmet and not missing_vest
        findings.append(
            Finding(
                "ppe",
                "ok" if ppe_ok else "violation",
                _max_confidence(detections, HELMET_LABELS | VEST_LABELS, default=0.5),
                frame,
                {
                    "person_count": person_count,
                    "helmet_count": helmet_count,
                    "vest_count": vest_count,
                    "missing_helmet": missing_helmet,
                    "missing_vest": missing_vest,
                    "tracked_people": ppe_summary.tracked_people
                    if ppe_summary is not None
                    else [],
                    "exempt_people": ppe_summary.exempt_people
                    if ppe_summary is not None
                    else [],
                },
                alert=not ppe_ok,
                severity="warning",
                message="Detected worker without safety helmet or reflective vest.",
            )
        )

    smoking_detected = bool(labels & SMOKING_LABELS)
    smoking_action = (
        activity_context.smoking_candidate if activity_context is not None else True
    )
    if smoking_detected and smoking_action:
        findings.append(
            Finding(
                "smoking",
                "detected",
                activity_context.smoking_confidence
                if activity_context is not None
                else _max_confidence(detections, SMOKING_LABELS),
                frame,
                {
                    "matched_labels": sorted(labels & SMOKING_LABELS),
                    "pose_summary": activity_context.pose_summary
                    if activity_context is not None
                    else {},
                },
                alert=True,
                severity="critical",
                message="Detected smoking behavior in construction area.",
            )
        )

    hot_work_detected = bool(labels & HOT_WORK_LABELS)
    hot_work_action = (
        activity_context.hot_work_candidate if activity_context is not None else True
    )
    if hot_work_detected and hot_work_action:
        findings.append(
            Finding(
                "hot_work",
                "detected",
                activity_context.hot_work_confidence
                if activity_context is not None
                else _max_confidence(detections, HOT_WORK_LABELS),
                frame,
                {
                    "matched_labels": sorted(labels & HOT_WORK_LABELS),
                    "pose_summary": activity_context.pose_summary
                    if activity_context is not None
                    else {},
                },
                alert=True,
                severity="critical",
                message="Detected hot work, sparks, flame, welding, or cutting.",
            )
        )

    return findings


def _max_confidence(
    detections: list[Detection], labels: set[str], default: float = 0.0
) -> float:
    values = [d.confidence for d in detections if d.label in labels]
    return max(values) if values else default
