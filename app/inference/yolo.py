from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config import Settings
from app.inference.base import Detection

logger = logging.getLogger(__name__)


class YoloDetector:
    """Small adapter around Ultralytics models.

    The service can run without model files so API integration and database
    flows remain testable before the GPU host is provisioned with weights.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.models: list[tuple[str, Any]] = []
        self.deep_sort: Any | None = None
        self._load_deepsort()
        self._load_model("seg", settings.yolo_seg_model)
        self._load_model("pose", settings.yolo_pose_model)
        if self.models:
            logger.info("loaded %d YOLO model(s)", len(self.models))
        else:
            logger.warning("no YOLO weights loaded; detection results will be empty")

    def _load_model(self, kind: str, path: str) -> None:
        if not Path(path).exists():
            logger.warning("YOLO %s model not found: %s", kind, path)
            return
        try:
            from ultralytics import YOLO

            self.models.append((kind, YOLO(path)))
            logger.info("loaded YOLO %s model from %s", kind, path)
        except Exception as exc:
            logger.exception("failed to load YOLO %s model from %s: %s", kind, path, exc)
            return

    def _load_deepsort(self) -> None:
        if self.settings.tracker_backend.lower() != "deepsort":
            return
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort

            self.deep_sort = DeepSort(max_age=30, n_init=2)
            logger.info("loaded DeepSORT tracker")
        except Exception as exc:
            logger.warning("DeepSORT unavailable; using IoU tracker fallback: %s", exc)

    @property
    def ready(self) -> bool:
        return bool(self.models)

    def detect(self, frame: Any) -> list[Detection]:
        detections: list[Detection] = []
        for kind, model in self.models:
            results = self._run_model(model, frame)
            for result in results:
                names = result.names or {}
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                keypoints = getattr(result, "keypoints", None)
                for box_index, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    label = str(names.get(cls_id, cls_id)).lower().replace(" ", "_")
                    confidence = float(box.conf[0])
                    xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
                    track_id = None
                    if getattr(box, "id", None) is not None:
                        track_id = int(box.id[0])
                    metadata: dict[str, Any] = {}
                    if keypoints is not None and getattr(keypoints, "data", None) is not None:
                        try:
                            metadata["keypoints"] = [
                                tuple(float(v) for v in point[:3])
                                for point in keypoints.data[box_index].tolist()
                            ]
                        except Exception:
                            metadata["keypoints"] = []
                    detections.append(
                        Detection(
                            label=label,
                            confidence=confidence,
                            bbox=xyxy,
                            track_id=track_id,
                            source=f"yolo_{kind}",
                            metadata=metadata,
                        )
                    )
        if self.settings.tracker_backend.lower() == "deepsort":
            detections = self._apply_deepsort(frame, detections)
        return detections

    def _run_model(self, model: Any, frame: Any) -> Any:
        if self.settings.tracker_backend.lower() == "bytetrack":
            try:
                return model.track(
                    frame,
                    conf=self.settings.detection_confidence,
                    tracker="bytetrack.yaml",
                    persist=True,
                    verbose=False,
                )
            except Exception as exc:
                logger.warning("ByteTrack failed, falling back to predict: %s", exc)
        elif self.settings.tracker_backend.lower() == "deepsort":
            logger.debug("using YOLO predict output before DeepSORT update")
        return model.predict(
            frame,
            conf=self.settings.detection_confidence,
            verbose=False,
        )

    def _apply_deepsort(self, frame: Any, detections: list[Detection]) -> list[Detection]:
        if self.deep_sort is None:
            return detections
        raw_tracks = []
        person_indices = []
        for index, detection in enumerate(detections):
            if detection.label not in {"person", "worker"} or detection.bbox is None:
                continue
            x1, y1, x2, y2 = detection.bbox
            raw_tracks.append(
                ([x1, y1, x2 - x1, y2 - y1], detection.confidence, detection.label)
            )
            person_indices.append(index)
        if not raw_tracks:
            return detections
        try:
            tracks = self.deep_sort.update_tracks(raw_tracks, frame=frame)
        except Exception as exc:
            logger.warning("DeepSORT update failed; keeping detections untracked: %s", exc)
            return detections

        updated = list(detections)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_box = tuple(float(v) for v in track.to_ltrb())
            best_index = _best_matching_detection(track_box, updated, person_indices)
            if best_index is None:
                continue
            detection = updated[best_index]
            updated[best_index] = Detection(
                label=detection.label,
                confidence=detection.confidence,
                bbox=detection.bbox,
                track_id=int(track.track_id),
                source=f"{detection.source}_deepsort",
                metadata=detection.metadata,
            )
        return updated


def _best_matching_detection(
    track_box: tuple[float, float, float, float],
    detections: list[Detection],
    indices: list[int],
) -> int | None:
    best_index = None
    best_iou = 0.0
    for index in indices:
        bbox = detections[index].bbox
        if bbox is None:
            continue
        value = _iou(track_box, bbox)
        if value > best_iou:
            best_iou = value
            best_index = index
    return best_index if best_iou >= 0.1 else None


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0
