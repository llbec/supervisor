from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.inference.base import Detection, FrameContext
from app.labels import HELMET_LABELS, HOT_WORK_LABELS, SMOKING_LABELS, VEST_LABELS
from app.tracking import PPESummary


@dataclass(frozen=True)
class Finding:
    event_type: str
    value: str
    confidence: float
    frame: FrameContext
    details: dict[str, Any]
    alert: bool = False
    severity: str = "warning"
    message: str | None = None


def summarize_realtime_frame(
    frame: FrameContext,
    detections: list[Detection],
    ppe: PPESummary,
    activity: dict[str, Any],
) -> list[Finding]:
    findings: list[Finding] = []
    labels = {d.label for d in detections}
    if ppe.person_count:
        ppe_ok = not ppe.missing_helmet and not ppe.missing_vest
        findings.append(
            Finding(
                "ppe",
                "ok" if ppe_ok else "violation",
                _max_confidence(detections, HELMET_LABELS | VEST_LABELS, 0.5),
                frame,
                {
                    "person_count": ppe.person_count,
                    "helmet_count": ppe.helmet_count,
                    "vest_count": ppe.vest_count,
                    "missing_helmet": ppe.missing_helmet,
                    "missing_vest": ppe.missing_vest,
                    "tracked_people": ppe.tracked_people,
                    "exempt_people": ppe.exempt_people,
                },
                alert=not ppe_ok,
                severity="warning",
                message="Detected worker without safety helmet or reflective vest.",
            )
        )
    if labels & SMOKING_LABELS and activity.get("smoking_candidate"):
        findings.append(
            Finding(
                "smoking",
                "detected",
                float(activity.get("smoking_confidence", 0.0)),
                frame,
                {
                    "matched_labels": sorted(labels & SMOKING_LABELS),
                    "pose_signals": activity.get("pose_signals", []),
                },
                alert=True,
                severity="critical",
                message="Detected smoking behavior in construction area.",
            )
        )
    if labels & HOT_WORK_LABELS and activity.get("hot_work_candidate"):
        findings.append(
            Finding(
                "hot_work",
                "detected",
                float(activity.get("hot_work_confidence", 0.0)),
                frame,
                {
                    "matched_labels": sorted(labels & HOT_WORK_LABELS),
                    "pose_signals": activity.get("pose_signals", []),
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
