from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.alerts import AlertPublisher
from app.config import Settings
from app.inference.base import FrameContext
from app.inference.qwen import VisionLanguageVerifier
from app.inference.yolo import YoloDetector
from app.models import AlertEvent, DetectionEvent, VideoJob
from app.rules import Finding, summarize_frame


class VideoProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.detector = YoloDetector(settings)
        self.verifier = VisionLanguageVerifier(settings)
        self.publisher = AlertPublisher(settings)

    def process_job(self, db: Session, job_id: int) -> None:
        job = db.get(VideoJob, job_id)
        if job is None:
            return

        job.status = "running"
        job.started_at = datetime.utcnow()
        db.commit()

        try:
            self._process_source(db, job)
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = datetime.utcnow()
            db.commit()
            return

        job.status = "completed" if job.source_type == "offline" else "running"
        job.finished_at = datetime.utcnow() if job.source_type == "offline" else None
        db.commit()

    def _process_source(self, db: Session, job: VideoJob) -> None:
        import cv2

        if job.source_type == "offline" and not Path(job.source).exists():
            raise FileNotFoundError(f"video not found: {job.source}")

        capture = cv2.VideoCapture(job.source)
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video source: {job.source}")

        frame_index = 0
        fps = capture.get(cv2.CAP_PROP_FPS) or 25
        max_stream_frames = None if job.source_type == "offline" else 10_000

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % self.settings.frame_sample_interval == 0:
                    context = FrameContext(
                        frame_index=frame_index,
                        timestamp_ms=int(frame_index * 1000 / fps),
                        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or frame.shape[1]),
                        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame.shape[0]),
                    )
                    self._process_frame(db, job, frame, context)
                frame_index += 1
                if max_stream_frames is not None and frame_index >= max_stream_frames:
                    break
        finally:
            capture.release()

    def _process_frame(self, db: Session, job: VideoJob, frame, context: FrameContext) -> None:
        detections = self.detector.detect(frame)
        scene = self.verifier.classify_scene(frame, detections)
        briefing = self.verifier.confirm_briefing(frame, detections)
        height_work = self.verifier.confirm_height_work(frame, detections)
        for finding in summarize_frame(context, detections, scene, briefing, height_work):
            self._store_finding(db, job, finding)
        db.commit()

    def _store_finding(self, db: Session, job: VideoJob, finding: Finding) -> None:
        event = DetectionEvent(
            job_id=job.id,
            camera_id=job.camera_id,
            event_type=finding.event_type,
            value=finding.value,
            confidence=finding.confidence,
            timestamp_ms=finding.frame.timestamp_ms,
            frame_index=finding.frame.frame_index,
            details=finding.details,
        )
        db.add(event)

        if not finding.alert:
            return

        alert = AlertEvent(
            job_id=job.id,
            camera_id=job.camera_id,
            alert_type=finding.event_type,
            severity=finding.severity,
            message=finding.message or finding.event_type,
            timestamp_ms=finding.frame.timestamp_ms,
            frame_index=finding.frame.frame_index,
            details=finding.details,
        )
        db.add(alert)
        db.flush()
        self.publisher.publish(alert)
