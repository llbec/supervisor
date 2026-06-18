from __future__ import annotations

from app.inference.base import Detection, FrameContext
from app.rules import summarize_frame
from app.tracking import PPETracker


class _Settings:
    ppe_required_hits = 1
    ppe_missing_tolerance = 0
    ppe_edge_margin_ratio = 0.03
    tracker_iou_threshold = 0.25
    frame_sample_interval = 15


def test_ppe_violation_generates_alert():
    frame = FrameContext(frame_index=0, timestamp_ms=0, width=1920, height=1080)
    findings = summarize_frame(
        frame,
        [Detection("person", 0.9), Detection("helmet", 0.8)],
        ("other", 0.4),
        (False, 0.2),
        (False, 0.2),
    )

    ppe = [finding for finding in findings if finding.event_type == "ppe"][0]
    assert ppe.value == "violation"
    assert ppe.alert is True


def test_smoking_and_hot_work_are_alerts():
    frame = FrameContext(frame_index=1, timestamp_ms=40, width=1920, height=1080)
    findings = summarize_frame(
        frame,
        [Detection("smoking", 0.7), Detection("spark", 0.85)],
        ("other", 0.4),
        (False, 0.2),
        (False, 0.2),
    )

    alert_types = {finding.event_type for finding in findings if finding.alert}
    assert {"smoking", "hot_work"} <= alert_types


def test_tracked_ppe_association_marks_worker_ok():
    tracker = PPETracker(_Settings())
    frame = FrameContext(frame_index=0, timestamp_ms=0, width=1920, height=1080)
    detections = [
        Detection("person", 0.9, (0, 0, 100, 200), track_id=12),
        Detection("helmet", 0.8, (20, 0, 70, 40)),
        Detection("vest", 0.8, (15, 70, 85, 150)),
    ]

    ppe_summary = tracker.update(detections, frame.frame_index)
    findings = summarize_frame(
        frame,
        detections,
        ("other", 0.4),
        (False, 0.2),
        (False, 0.2),
        ppe_summary,
    )

    ppe = [finding for finding in findings if finding.event_type == "ppe"][0]
    assert ppe.value == "ok"
    assert ppe.alert is False
    assert ppe.details["tracked_people"][0]["track_id"] == 12
