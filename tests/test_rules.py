from __future__ import annotations

from app.inference.base import Detection, FrameContext
from app.rules import summarize_frame


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
