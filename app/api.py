from __future__ import annotations

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db, init_db
from app.logging_config import configure_logging
from app.models import AlertEvent, DetectionEvent, VideoJob
from app.processor import VideoProcessor
from app.schemas import (
    AlertResponse,
    EventResponse,
    JobResponse,
    OfflineVideoRequest,
    StreamRequest,
)

app = FastAPI(title="Supervisor Agent", version="0.1.0")


@app.on_event("startup")
def startup() -> None:
    configure_logging(get_settings().log_level)
    init_db()


def run_job(job_id: int) -> None:
    from app.db import SessionLocal

    db = SessionLocal()
    try:
        VideoProcessor(get_settings()).process_job(db, job_id)
    finally:
        db.close()


@app.post("/videos/offline", response_model=JobResponse)
def submit_offline_video(
    request: OfflineVideoRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> VideoJob:
    job = VideoJob(source=request.path, source_type="offline", camera_id=request.camera_id)
    db.add(job)
    db.commit()
    db.refresh(job)
    background_tasks.add_task(run_job, job.id)
    return job


@app.post("/streams", response_model=JobResponse)
def submit_stream(
    request: StreamRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> VideoJob:
    job = VideoJob(source=request.url, source_type="stream", camera_id=request.camera_id)
    db.add(job)
    db.commit()
    db.refresh(job)
    background_tasks.add_task(run_job, job.id)
    return job


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: int, db: Session = Depends(get_db)) -> VideoJob:
    job = db.get(VideoJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/jobs", response_model=list[JobResponse])
def list_jobs(limit: int = 50, db: Session = Depends(get_db)) -> list[VideoJob]:
    return list(db.scalars(select(VideoJob).order_by(desc(VideoJob.id)).limit(limit)))


@app.get("/events", response_model=list[EventResponse])
def list_events(
    job_id: int | None = None,
    event_type: str | None = None,
    limit: int = 200,
    db: Session = Depends(get_db),
) -> list[DetectionEvent]:
    stmt = select(DetectionEvent).order_by(desc(DetectionEvent.id)).limit(limit)
    if job_id is not None:
        stmt = stmt.where(DetectionEvent.job_id == job_id)
    if event_type is not None:
        stmt = stmt.where(DetectionEvent.event_type == event_type)
    return list(db.scalars(stmt))


@app.get("/alerts", response_model=list[AlertResponse])
def list_alerts(
    job_id: int | None = None,
    limit: int = 200,
    db: Session = Depends(get_db),
) -> list[AlertEvent]:
    stmt = select(AlertEvent).order_by(desc(AlertEvent.id)).limit(limit)
    if job_id is not None:
        stmt = stmt.where(AlertEvent.job_id == job_id)
    return list(db.scalars(stmt))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
