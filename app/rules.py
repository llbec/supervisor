from __future__ import annotations

from dataclasses import dataclass

from app.inference.base import Detection, FrameContext


HELMET_LABELS = {"helmet", "hardhat", "safety_helmet", "head_helmet"}
VEST_LABELS = {"vest", "safety_vest", "reflective_vest", "hi_vis_vest"}
SMOKING_LABELS = {"smoke", "smoking", "cigarette"}
HOT_WORK_LABELS = {
    "fire",
    "flame",
    "spark",
    "sparks",
    "welding",
    "welder",
    "cutting",
    "open_flame",
}


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


def summarize_frame(
    frame: FrameContext,
    detections: list[Detection],
    scene: tuple[str, float],
    briefing: tuple[bool, float],
    height_work: tuple[bool, float],
) -> list[Finding]:
    findings: list[Finding] = []
    labels = {d.label for d in detections}

    findings.append(
        Finding("scene", scene[0], scene[1], frame, {"labels": sorted(labels)})
    )
    findings.append(
        Finding(
            "height_work",
            str(height_work[0]).lower(),
            height_work[1],
            frame,
            {"labels": sorted(labels)},
        )
    )
    findings.append(
        Finding(
            "briefing",
            str(briefing[0]).lower(),
            briefing[1],
            frame,
            {"person_count": sum(1 for d in detections if d.label == "person")},
        )
    )

    person_count = sum(1 for d in detections if d.label == "person")
    helmet_count = sum(1 for d in detections if d.label in HELMET_LABELS)
    vest_count = sum(1 for d in detections if d.label in VEST_LABELS)
    if person_count:
        missing_helmet = helmet_count < person_count
        missing_vest = vest_count < person_count
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
                },
                alert=not ppe_ok,
                severity="warning",
                message="Detected worker without safety helmet or reflective vest.",
            )
        )

    if labels & SMOKING_LABELS:
        findings.append(
            Finding(
                "smoking",
                "detected",
                _max_confidence(detections, SMOKING_LABELS),
                frame,
                {"matched_labels": sorted(labels & SMOKING_LABELS)},
                alert=True,
                severity="critical",
                message="Detected smoking behavior in construction area.",
            )
        )

    if labels & HOT_WORK_LABELS:
        findings.append(
            Finding(
                "hot_work",
                "detected",
                _max_confidence(detections, HOT_WORK_LABELS),
                frame,
                {"matched_labels": sorted(labels & HOT_WORK_LABELS)},
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
