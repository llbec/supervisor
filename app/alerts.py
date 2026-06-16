from __future__ import annotations

import httpx

from app.config import Settings
from app.models import AlertEvent


class AlertPublisher:
    def __init__(self, settings: Settings):
        self.settings = settings

    def publish(self, alert: AlertEvent) -> None:
        if not self.settings.alert_webhook_url:
            return
        payload = {
            "id": alert.id,
            "job_id": alert.job_id,
            "camera_id": alert.camera_id,
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "message": alert.message,
            "timestamp_ms": alert.timestamp_ms,
            "frame_index": alert.frame_index,
            "details": alert.details,
        }
        try:
            httpx.post(self.settings.alert_webhook_url, json=payload, timeout=3.0)
        except httpx.HTTPError:
            return
