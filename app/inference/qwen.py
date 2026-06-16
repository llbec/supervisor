from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings
from app.inference.base import Detection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QwenAnalysis:
    scene: str
    scene_confidence: float
    briefing: bool
    briefing_confidence: float
    height_work: bool
    height_work_confidence: float


class VisionLanguageVerifier:
    """Qwen-VL verifier with a conservative rule fallback."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.processor: Any | None = None
        self.model: Any | None = None
        self._last_frame_id: int | None = None
        self._last_analysis: QwenAnalysis | None = None
        if settings.qwen_enabled:
            self._load_qwen()
        else:
            logger.info("Qwen verifier disabled by configuration")

    @property
    def ready(self) -> bool:
        return self.processor is not None and self.model is not None

    def classify_scene(self, frame: Any, detections: list[Detection]) -> tuple[str, float]:
        analysis = self._analyze(frame, detections)
        return analysis.scene, analysis.scene_confidence

    def confirm_briefing(self, frame: Any, detections: list[Detection]) -> tuple[bool, float]:
        analysis = self._analyze(frame, detections)
        return analysis.briefing, analysis.briefing_confidence

    def confirm_height_work(self, frame: Any, detections: list[Detection]) -> tuple[bool, float]:
        analysis = self._analyze(frame, detections)
        return analysis.height_work, analysis.height_work_confidence

    def _load_qwen(self) -> None:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception as exc:
            logger.warning("Qwen dependencies are unavailable, using rule fallback: %s", exc)
            return

        try:
            model_path = self._resolve_qwen_model_path()
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                device_map=self.settings.qwen_device_map,
                torch_dtype="auto",
                trust_remote_code=True,
            )
            logger.info("loaded Qwen-VL verifier: %s", model_path)
        except Exception as exc:
            self.processor = None
            self.model = None
            logger.exception("failed to load Qwen-VL verifier, using rule fallback: %s", exc)

    def _resolve_qwen_model_path(self) -> str:
        model_id = self.settings.qwen_model
        if Path(model_id).exists():
            logger.info("using local Qwen model path: %s", model_id)
            return model_id
        if not self.settings.qwen_use_modelscope:
            logger.info("using Qwen model from Transformers source: %s", model_id)
            return model_id

        modelscope_id = self.settings.qwen_modelscope_model or model_id
        try:
            from modelscope import snapshot_download
        except Exception as exc:
            logger.warning(
                "ModelScope is unavailable, using Transformers source %s: %s",
                model_id,
                exc,
            )
            return model_id

        try:
            kwargs: dict[str, Any] = {"model_id": modelscope_id}
            if self.settings.qwen_cache_dir:
                kwargs["cache_dir"] = self.settings.qwen_cache_dir
            if self.settings.qwen_modelscope_revision:
                kwargs["revision"] = self.settings.qwen_modelscope_revision
            local_path = snapshot_download(**kwargs)
            logger.info(
                "downloaded/resolved Qwen model from ModelScope model_id=%s path=%s",
                modelscope_id,
                local_path,
            )
            return str(local_path)
        except Exception as exc:
            logger.exception(
                "failed to download Qwen model from ModelScope model_id=%s, "
                "using Transformers source %s: %s",
                modelscope_id,
                model_id,
                exc,
            )
            return model_id

    def _analyze(self, frame: Any, detections: list[Detection]) -> QwenAnalysis:
        frame_id = id(frame)
        if self._last_frame_id == frame_id and self._last_analysis is not None:
            return self._last_analysis

        if self.ready:
            analysis = self._analyze_with_qwen(frame, detections)
        else:
            analysis = self._fallback_analysis(detections)

        self._last_frame_id = frame_id
        self._last_analysis = analysis
        return analysis

    def _analyze_with_qwen(
        self, frame: Any, detections: list[Detection]
    ) -> QwenAnalysis:
        assert self.processor is not None
        assert self.model is not None

        image = _frame_to_pil(frame)
        labels = _format_detections(detections)
        prompt = (
            "你是工地施工安全视频审核模型。请根据图片和YOLO检测标签判断："
            "1) scene 只能是 machine_room、near_tower、other；"
            "2) briefing 是否存在一人对多人进行任务交底或安全强调；"
            "3) height_work 是否存在登高作业。"
            "只输出JSON，不要解释。格式："
            '{"scene":"other","scene_confidence":0.0,'
            '"briefing":false,"briefing_confidence":0.0,'
            '"height_work":false,"height_work_confidence":0.0}'
            f"\nYOLO检测标签：{labels}"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            if hasattr(inputs, "to") and hasattr(self.model, "device"):
                inputs = inputs.to(self.model.device)
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.settings.qwen_max_new_tokens,
            )
            input_length = inputs["input_ids"].shape[-1]
            output_ids = generated_ids[:, input_length:]
            output = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            parsed = _parse_json(output)
            return QwenAnalysis(
                scene=_normalize_scene(parsed.get("scene")),
                scene_confidence=_confidence(parsed.get("scene_confidence")),
                briefing=_bool(parsed.get("briefing")),
                briefing_confidence=_confidence(parsed.get("briefing_confidence")),
                height_work=_bool(parsed.get("height_work")),
                height_work_confidence=_confidence(parsed.get("height_work_confidence")),
            )
        except Exception as exc:
            logger.exception("Qwen-VL inference failed, using rule fallback: %s", exc)
            return self._fallback_analysis(detections)

    def _fallback_analysis(self, detections: list[Detection]) -> QwenAnalysis:
        labels = {d.label for d in detections}
        if labels & {"server_rack", "cabinet", "machine_room", "equipment_room"}:
            scene = "machine_room"
            scene_confidence = 0.75
        elif labels & {"tower", "telecom_tower", "pylon"}:
            scene = "near_tower"
            scene_confidence = 0.75
        else:
            scene = "other"
            scene_confidence = 0.4

        person_count = sum(1 for d in detections if d.label in {"person", "worker"})
        briefing = person_count >= 3
        height_work = bool(
            labels & {"ladder", "scaffold", "tower", "telecom_tower", "aerial_lift"}
        )
        return QwenAnalysis(
            scene=scene,
            scene_confidence=scene_confidence,
            briefing=briefing,
            briefing_confidence=0.45 if briefing else 0.2,
            height_work=height_work,
            height_work_confidence=0.7 if height_work else 0.25,
        )


def _frame_to_pil(frame: Any) -> Any:
    from PIL import Image

    try:
        import cv2

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return Image.fromarray(frame)


def _format_detections(detections: list[Detection]) -> list[dict[str, Any]]:
    return [
        {
            "label": detection.label,
            "confidence": round(detection.confidence, 3),
            "bbox": detection.bbox,
            "track_id": detection.track_id,
        }
        for detection in detections
    ]


def _parse_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"Qwen output is not JSON: {text}")
    return json.loads(match.group(0))


def _normalize_scene(value: Any) -> str:
    if value in {"machine_room", "near_tower", "other"}:
        return str(value)
    return "other"


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "是", "有"}
    return bool(value)
