from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import Settings
from app.inference.base import Detection


class YoloDetector:
    """Small adapter around Ultralytics models.

    The service can run without model files so API integration and database
    flows remain testable before the GPU host is provisioned with weights.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.models: list[tuple[str, Any]] = []
        self._load_model("seg", settings.yolo_seg_model)
        self._load_model("pose", settings.yolo_pose_model)

    def _load_model(self, kind: str, path: str) -> None:
        if not Path(path).exists():
            return
        try:
            from ultralytics import YOLO

            self.models.append((kind, YOLO(path)))
        except Exception:
            return

    @property
    def ready(self) -> bool:
        return bool(self.models)

    def detect(self, frame: Any) -> list[Detection]:
        detections: list[Detection] = []
        for kind, model in self.models:
            results = model.predict(
                frame,
                conf=self.settings.detection_confidence,
                verbose=False,
            )
            for result in results:
                names = result.names or {}
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = str(names.get(cls_id, cls_id)).lower().replace(" ", "_")
                    confidence = float(box.conf[0])
                    xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
                    detections.append(
                        Detection(
                            label=label,
                            confidence=confidence,
                            bbox=xyxy,
                            source=f"yolo_{kind}",
                        )
                    )
        return detections
