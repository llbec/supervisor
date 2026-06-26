from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.activity import ActivityAnalyzer, CandidateSegment
from app.alerts import AlertPublisher
from app.config import Settings
from app.inference.base import FrameContext
from app.inference.qwen import MultimodalClient
from app.inference.yolo import YoloDetector
from app.models import AlertEvent, DetectionEvent, VideoJob
from app.rules import Finding, summarize_realtime_frame
from app.scene import SceneAggregator
from app.tracking import PPETracker

logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.detector = YoloDetector(settings)
        self.ppe_tracker = PPETracker(settings)
        self.activity = ActivityAnalyzer(settings)
        self.scene = SceneAggregator()
        self.multimodal = MultimodalClient(settings)
        self.publisher = AlertPublisher(settings)

    def process_job(self, db: Session, job_id: int) -> None:
        job = db.get(VideoJob, job_id)
        if job is None:
            logger.warning("job not found: %s", job_id)
            return
        job.status = "running"
        job.started_at = datetime.now()
        db.commit()
        try:
            self._process_source(db, job)
        except Exception as exc:
            logger.exception("job %s failed: %s", job.id, exc)
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = datetime.now()
            db.commit()
            return
        job.status = "completed" if job.source_type == "offline" else "running"
        job.finished_at = datetime.now() if job.source_type == "offline" else None
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
            self._finalize(db, job)
        finally:
            capture.release()

    def _process_frame(self, db: Session, job: VideoJob, frame, context: FrameContext) -> None:
        detections = self.detector.detect(frame)
        self.scene.update(frame, context, detections)
        ppe = self.ppe_tracker.update(detections, context.frame_index, context.width, context.height)
        activity = self.activity.update(context, detections)
        for finding in summarize_realtime_frame(context, detections, ppe, activity):
            self._store_finding(db, job, finding)
        db.commit()

    def _finalize(self, db: Session, job: VideoJob) -> None:
        self.activity.finalize()
        representative = self.scene.representative
        if representative is None:
            logger.warning("job %s has no representative frame", job.id)
            return
        snapshot = self._save_snapshot(job, representative.frame, representative.context, "scene")

        scene_result = self.multimodal.analyze_scene(
            representative.frame,
            {**self.scene.summary(), **self.activity.summary()},
        )
        self._store_final_event(
            db,
            job,
            "scene",
            str(scene_result.get("scene", "other")),
            float(scene_result.get("scene_confidence", 0.0)),
            representative.context,
            {**scene_result, "snapshot_path": snapshot},
        )

        height_segments = [segment.to_dict() for segment in self.activity.height_candidates]
        height_result = self.multimodal.analyze_height_work(representative.frame, height_segments)
        self._store_candidate_results(db, job, "height_work", representative.context, height_result.raw, snapshot)

        briefing_segments = [segment.to_dict() for segment in self.activity.briefing_candidates]
        briefing_result = self.multimodal.analyze_briefing(representative.frame, briefing_segments)
        self._store_candidate_results(db, job, "briefing", representative.context, briefing_result.raw, snapshot)
        db.commit()

    def _store_candidate_results(
        self,
        db: Session,
        job: VideoJob,
        event_type: str,
        context: FrameContext,
        raw: dict[str, Any],
        snapshot: str | None,
    ) -> None:
        candidates = raw.get("confirmed_candidates", [])
        if not candidates:
            self._store_final_event(
                db,
                job,
                event_type,
                "false",
                0.0,
                context,
                {**raw, "snapshot_path": snapshot},
            )
            return
        for candidate in candidates:
            confirmed = bool(candidate.get("confirmed", False))
            self._store_final_event(
                db,
                job,
                event_type,
                str(confirmed).lower(),
                float(candidate.get("confidence", 0.0)),
                FrameContext(
                    frame_index=context.frame_index,
                    timestamp_ms=int(candidate.get("start_ms") or context.timestamp_ms),
                    width=context.width,
                    height=context.height,
                ),
                {**candidate, "snapshot_path": snapshot},
            )

    def _store_final_event(
        self,
        db: Session,
        job: VideoJob,
        event_type: str,
        value: str,
        confidence: float,
        context: FrameContext,
        details: dict[str, Any],
    ) -> None:
        db.add(
            DetectionEvent(
                job_id=job.id,
                camera_id=job.camera_id,
                event_type=event_type,
                value=value,
                confidence=confidence,
                timestamp_ms=context.timestamp_ms,
                frame_index=context.frame_index,
                details=details,
            )
        )

    def _store_finding(self, db: Session, job: VideoJob, finding: Finding) -> None:
        db.add(
            DetectionEvent(
                job_id=job.id,
                camera_id=job.camera_id,
                event_type=finding.event_type,
                value=finding.value,
                confidence=finding.confidence,
                timestamp_ms=finding.frame.timestamp_ms,
                frame_index=finding.frame.frame_index,
                details=finding.details,
            )
        )
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

    def _save_snapshot(self, job: VideoJob, frame, context: FrameContext, category: str) -> str | None:
        import cv2

        output_dir = Path(self.settings.snapshot_dir) / f"job_{job.id}" / category
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{context.timestamp_ms}_{context.frame_index}.jpg"
        if not cv2.imwrite(str(path), frame):
            return None
        return str(path)
