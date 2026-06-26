from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config import Settings
from app.inference.base import Detection

logger = logging.getLogger(__name__)


class YoloDetector:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.models: list[tuple[str, Any]] = []
        self._load_model("seg", settings.yolo_seg_model)
        self._load_model("pose", settings.yolo_pose_model)
        if not self.models:
            logger.warning("no YOLO weights loaded; detection results will be empty")

    def _load_model(self, kind: str, path: str) -> None:
        if not Path(path).exists():
            logger.warning("YOLO %s model not found: %s", kind, path)
            return
        try:
            from ultralytics import YOLO

            self.models.append((kind, YOLO(path)))
            logger.info("loaded YOLO %s model: %s", kind, path)
        except Exception as exc:
            logger.exception("failed to load YOLO %s model %s: %s", kind, path, exc)

    def detect(self, frame: Any) -> list[Detection]:
        detections: list[Detection] = []
        for kind, model in self.models:
            results = self._run_model(model, frame)
            for result in results:
                detections.extend(_detections_from_result(kind, result))
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
                logger.warning("ByteTrack failed; falling back to predict: %s", exc)
        return model.predict(
            frame,
            conf=self.settings.detection_confidence,
            verbose=False,
        )


def _detections_from_result(kind: str, result: Any) -> list[Detection]:
    detections: list[Detection] = []
    names = result.names or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections
    keypoints = getattr(result, "keypoints", None)
    for index, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        label = str(names.get(cls_id, cls_id)).lower().replace(" ", "_")
        confidence = float(box.conf[0])
        bbox = tuple(float(v) for v in box.xyxy[0].tolist())
        track_id = None
        if getattr(box, "id", None) is not None:
            track_id = int(box.id[0])
        metadata: dict[str, Any] = {}
        if keypoints is not None and getattr(keypoints, "data", None) is not None:
            try:
                metadata["keypoints"] = [
                    tuple(float(v) for v in point[:3])
                    for point in keypoints.data[index].tolist()
                ]
            except Exception:
                metadata["keypoints"] = []
        detections.append(
            Detection(
                label=label,
                confidence=confidence,
                bbox=bbox,
                track_id=track_id,
                source=f"yolo_{kind}",
                metadata=metadata,
            )
        )
    return detections
