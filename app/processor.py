from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.activity import TrajectoryBuffer
from app.alerts import AlertPublisher
from app.config import Settings
from app.inference.base import Detection, FrameContext
from app.labels import HOT_WORK_LABELS, SMOKING_LABELS
from app.inference.qwen import VisionLanguageVerifier
from app.inference.yolo import YoloDetector
from app.models import AlertEvent, DetectionEvent, VideoJob
from app.rules import Finding, summarize_frame
from app.scene import SceneSampler
from app.tracking import PPETracker

logger = logging.getLogger(__name__)


@dataclass
class RepresentativeFrame:
    frame: Any
    context: FrameContext
    detections: list[Detection]
    score: int


@dataclass
class VideoAggregate:
    sampled_frames: int = 0
    label_counts: Counter[str] = field(default_factory=Counter)
    max_confidence: dict[str, float] = field(default_factory=dict)
    scene_signatures: Counter[tuple[str, ...]] = field(default_factory=Counter)
    ppe_violations: list[dict[str, Any]] = field(default_factory=list)
    smoking_candidates: list[dict[str, Any]] = field(default_factory=list)
    hot_work_candidates: list[dict[str, Any]] = field(default_factory=list)
    representative: RepresentativeFrame | None = None


class VideoProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.detector = YoloDetector(settings)
        self.verifier = VisionLanguageVerifier(settings)
        self.publisher = AlertPublisher(settings)
        self.ppe_tracker = PPETracker(settings)
        self.scene_sampler = SceneSampler(settings)
        self.trajectory_buffer = TrajectoryBuffer(settings)

    def process_job(self, db: Session, job_id: int) -> None:
        job = db.get(VideoJob, job_id)
        if job is None:
            logger.warning("job not found: %s", job_id)
            return

        logger.info(
            "job %s started source_type=%s camera_id=%s source=%s",
            job.id,
            job.source_type,
            job.camera_id,
            job.source,
        )
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
        logger.info("job %s finished status=%s", job.id, job.status)

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
        aggregate = VideoAggregate()
        logger.info("opened video source job_id=%s fps=%.2f", job.id, fps)

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
                    self._process_yolo_frame(db, job, frame, context, aggregate)
                    if frame_index % (self.settings.frame_sample_interval * 20) == 0:
                        logger.info(
                            "job %s processed frame=%s timestamp_ms=%s",
                            job.id,
                            frame_index,
                            context.timestamp_ms,
                        )
                frame_index += 1
                if max_stream_frames is not None and frame_index >= max_stream_frames:
                    break
            self._finalize_multimodal_analysis(db, job, aggregate)
        finally:
            capture.release()

    def _process_yolo_frame(
        self,
        db: Session,
        job: VideoJob,
        frame,
        context: FrameContext,
        aggregate: VideoAggregate,
    ) -> None:
        detections = self.detector.detect(frame)
        self._update_aggregate(aggregate, frame, context, detections)
        ppe_summary = self.ppe_tracker.update(
            detections,
            context.frame_index,
            frame_width=context.width,
            frame_height=context.height,
        )
        activity_context = self.trajectory_buffer.update(context, detections)

        scene = ("unknown", 0.0)
        scene_candidate = self.scene_sampler.update(context, detections)
        if scene_candidate is not None:
            aggregate.scene_signatures[scene_candidate.signature] += 1
            logger.info(
                "job %s collected scene candidate frame=%s signature=%s",
                job.id,
                context.frame_index,
                scene_candidate.signature,
            )

        briefing = (False, 0.0)
        height_work = (False, 0.0)

        logger.debug(
            "job %s frame=%s detections=%s scene=%s briefing=%s height_work=%s ppe=%s",
            job.id,
            context.frame_index,
            len(detections),
            scene,
            briefing,
            height_work,
            ppe_summary,
        )
        self._collect_candidates(aggregate, context, detections, ppe_summary, activity_context)
        for finding in summarize_frame(
            context,
            detections,
            scene,
            briefing,
            height_work,
            ppe_summary,
            activity_context,
        ):
            self._store_finding(db, job, finding)
        db.commit()

    def _finalize_multimodal_analysis(
        self, db: Session, job: VideoJob, aggregate: VideoAggregate
    ) -> None:
        if aggregate.representative is None:
            logger.warning("job %s has no representative frame for Qwen analysis", job.id)
            return

        representative = aggregate.representative
        snapshot_path = self._save_snapshot(
            job, representative.frame, representative.context, "scene"
        )
        summary = self._build_multimodal_summary(aggregate)
        analysis = self.verifier.analyze_video_summary(
            representative.frame,
            summary,
            representative.detections,
        )
        logger.info(
            "job %s final multimodal analysis scene=%s height_work=%s briefing=%s reason=%s",
            job.id,
            analysis.scene,
            analysis.height_work,
            analysis.briefing,
            analysis.reason,
        )
        for finding in summarize_frame(
            representative.context,
            representative.detections,
            (analysis.scene, analysis.scene_confidence),
            (analysis.briefing, analysis.briefing_confidence),
            (analysis.height_work, analysis.height_work_confidence),
            activity_context=None,
            scene_snapshot=snapshot_path,
            activity_snapshot=snapshot_path,
            activity_reason=analysis.reason,
        ):
            if finding.event_type in {"scene", "height_work", "briefing"}:
                self._store_finding(db, job, finding)
        db.commit()

    def _update_aggregate(
        self,
        aggregate: VideoAggregate,
        frame,
        context: FrameContext,
        detections: list[Detection],
    ) -> None:
        aggregate.sampled_frames += 1
        for detection in detections:
            aggregate.label_counts[detection.label] += 1
            aggregate.max_confidence[detection.label] = max(
                aggregate.max_confidence.get(detection.label, 0.0),
                detection.confidence,
            )
        score = _representative_score(detections)
        if aggregate.representative is None or score > aggregate.representative.score:
            aggregate.representative = RepresentativeFrame(
                frame=frame.copy(),
                context=context,
                detections=list(detections),
                score=score,
            )

    def _collect_candidates(
        self,
        aggregate: VideoAggregate,
        context: FrameContext,
        detections: list[Detection],
        ppe_summary,
        activity_context,
    ) -> None:
        if ppe_summary is not None and (ppe_summary.missing_helmet or ppe_summary.missing_vest):
            aggregate.ppe_violations.append(
                {
                    "timestamp_ms": context.timestamp_ms,
                    "frame_index": context.frame_index,
                    "missing_helmet": ppe_summary.missing_helmet,
                    "missing_vest": ppe_summary.missing_vest,
                    "tracked_people": ppe_summary.tracked_people[:8],
                    "exempt_people": ppe_summary.exempt_people[:8],
                }
            )
        labels = {d.label for d in detections}
        if labels & SMOKING_LABELS:
            aggregate.smoking_candidates.append(
                {
                    "timestamp_ms": context.timestamp_ms,
                    "frame_index": context.frame_index,
                    "labels": sorted(labels & SMOKING_LABELS),
                    "with_pose_action": activity_context.smoking_candidate,
                    "confidence": activity_context.smoking_confidence,
                }
            )
        if labels & HOT_WORK_LABELS:
            aggregate.hot_work_candidates.append(
                {
                    "timestamp_ms": context.timestamp_ms,
                    "frame_index": context.frame_index,
                    "labels": sorted(labels & HOT_WORK_LABELS),
                    "with_pose_action": activity_context.hot_work_candidate,
                    "confidence": activity_context.hot_work_confidence,
                }
            )

    def _build_multimodal_summary(self, aggregate: VideoAggregate) -> dict[str, Any]:
        return {
            "sampled_frames": aggregate.sampled_frames,
            "top_labels": aggregate.label_counts.most_common(30),
            "max_confidence": dict(
                sorted(
                    aggregate.max_confidence.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:30]
            ),
            "scene_signatures": [
                {"labels": list(signature), "count": count}
                for signature, count in aggregate.scene_signatures.most_common(10)
            ],
            "ppe_violations": aggregate.ppe_violations[:50],
            "smoking_candidates": aggregate.smoking_candidates[:50],
            "hot_work_candidates": aggregate.hot_work_candidates[:50],
            "trajectory_pose_summary": self.trajectory_buffer.summary(),
        }

    def _save_snapshot(
        self, job: VideoJob, frame, context: FrameContext, category: str
    ) -> str | None:
        import cv2

        output_dir = Path(self.settings.snapshot_dir) / f"job_{job.id}" / category
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{context.timestamp_ms}_{context.frame_index}.jpg"
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            logger.warning("failed to save snapshot: %s", path)
            return None
        return str(path)

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
        logger.debug(
            "stored event job_id=%s type=%s value=%s confidence=%.3f frame=%s",
            job.id,
            finding.event_type,
            finding.value,
            finding.confidence,
            finding.frame.frame_index,
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
        logger.warning(
            "stored alert id=%s job_id=%s type=%s severity=%s frame=%s",
            alert.id,
            job.id,
            alert.alert_type,
            alert.severity,
            alert.frame_index,
        )
        self.publisher.publish(alert)


def _representative_score(detections: list[Detection]) -> int:
    labels = {detection.label for detection in detections}
    person_count = sum(1 for detection in detections if detection.label in {"person", "worker"})
    context_labels = labels - {"person", "worker", "helmet", "safety_helmet", "vest"}
    return len(context_labels) * 5 + person_count * 2 + len(detections)
