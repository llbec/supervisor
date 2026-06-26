from __future__ import annotations

from types import SimpleNamespace

from app.inference.base import Detection, FrameContext
from app.rules import summarize_realtime_frame
from app.tracking import PPETracker


class _Settings:
    ppe_required_hits = 1
    ppe_missing_tolerance = 0
    ppe_edge_margin_ratio = 0.03
    tracker_iou_threshold = 0.25


def test_ppe_violation_generates_alert():
    tracker = PPETracker(_Settings())
    frame = FrameContext(frame_index=0, timestamp_ms=0, width=1920, height=1080)
    detections = [Detection("person", 0.9, (100, 100, 300, 700), track_id=1)]

    ppe = tracker.update(detections, frame.frame_index, frame.width, frame.height)
    findings = summarize_realtime_frame(frame, detections, ppe, {})

    ppe_finding = [finding for finding in findings if finding.event_type == "ppe"][0]
    assert ppe_finding.value == "violation"
    assert ppe_finding.alert is True


def test_ppe_edge_exemption_suppresses_violation():
    tracker = PPETracker(_Settings())
    frame = FrameContext(frame_index=0, timestamp_ms=0, width=1000, height=1000)
    detections = [Detection("person", 0.9, (0, 0, 100, 200), track_id=1)]

    ppe = tracker.update(detections, frame.frame_index, frame.width, frame.height)
    findings = summarize_realtime_frame(frame, detections, ppe, {})

    ppe_finding = [finding for finding in findings if finding.event_type == "ppe"][0]
    assert ppe_finding.value == "ok"
    assert ppe.exempt_people


def test_smoking_and_hot_work_require_activity_signal():
    frame = FrameContext(frame_index=0, timestamp_ms=0, width=1920, height=1080)
    detections = [Detection("cigarette", 0.9), Detection("spark", 0.8)]
    ppe = SimpleNamespace(
        person_count=0,
        helmet_count=0,
        vest_count=0,
        missing_helmet=False,
        missing_vest=False,
        tracked_people=[],
        exempt_people=[],
    )
    findings = summarize_realtime_frame(
        frame,
        detections,
        ppe,
        {
            "smoking_candidate": False,
            "smoking_confidence": 0.5,
            "hot_work_candidate": True,
            "hot_work_confidence": 0.8,
            "pose_signals": ["work_pose"],
        },
    )

    alert_types = {finding.event_type for finding in findings if finding.alert}
    assert "smoking" not in alert_types
    assert "hot_work" in alert_types
