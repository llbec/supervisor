from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class OfflineVideoRequest(BaseModel):
    path: str = Field(..., description="Local video path visible to the service.")
    camera_id: str | None = None


class StreamRequest(BaseModel):
    url: str = Field(..., description="RTSP/RTMP/HTTP video stream URL.")
    camera_id: str | None = None


class JobResponse(BaseModel):
    id: int
    source: str
    source_type: str
    camera_id: str | None
    status: str
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    model_config = {"from_attributes": True}


class EventResponse(BaseModel):
    id: int
    job_id: int
    camera_id: str | None
    event_type: str
    value: str
    confidence: float
    timestamp_ms: int
    frame_index: int
    details: dict[str, Any]
    created_at: datetime

    model_config = {"from_attributes": True}


class AlertResponse(BaseModel):
    id: int
    job_id: int
    camera_id: str | None
    alert_type: str
    severity: str
    message: str
    timestamp_ms: int
    frame_index: int
    details: dict[str, Any]
    created_at: datetime

    model_config = {"from_attributes": True}
