from __future__ import annotations

import json
import logging
import re
import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx

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


@dataclass(frozen=True)
class TrackAnalysis:
    height_work: bool
    height_work_confidence: float
    briefing: bool
    briefing_confidence: float
    reason: str


@dataclass(frozen=True)
class VideoSummaryAnalysis:
    scene: str
    scene_confidence: float
    height_work: bool
    height_work_confidence: float
    briefing: bool
    briefing_confidence: float
    reason: str


class VisionLanguageVerifier:
    """Qwen-VL verifier with a conservative rule fallback."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.processor: Any | None = None
        self.model: Any | None = None
        self.remote_client: RemoteMultimodalClient | None = None
        self._last_frame_id: int | None = None
        self._last_analysis: QwenAnalysis | None = None
        if not settings.qwen_enabled:
            logger.info("multimodal verifier disabled by configuration")
        elif settings.multimodal_provider == "local":
            self._load_qwen()
        else:
            self.remote_client = RemoteMultimodalClient(settings)

    @property
    def ready(self) -> bool:
        return (
            self.remote_client is not None
            and self.remote_client.ready
            or self.processor is not None
            and self.model is not None
        )

    def classify_scene(self, frame: Any, detections: list[Detection]) -> tuple[str, float]:
        analysis = self._analyze(frame, detections)
        return analysis.scene, analysis.scene_confidence

    def confirm_briefing(self, frame: Any, detections: list[Detection]) -> tuple[bool, float]:
        analysis = self._analyze(frame, detections)
        return analysis.briefing, analysis.briefing_confidence

    def confirm_height_work(self, frame: Any, detections: list[Detection]) -> tuple[bool, float]:
        analysis = self._analyze(frame, detections)
        return analysis.height_work, analysis.height_work_confidence

    def analyze_scene(
        self, frame: Any, detections: list[Detection], signature: tuple[str, ...]
    ) -> tuple[str, float, str]:
        if not self.ready:
            fallback = self._fallback_analysis(detections)
            return fallback.scene, fallback.scene_confidence, "rule_fallback"
        prompt = (
            "你是工地施工视频场景分类模型。请根据图片和YOLO场景标签判断场景。"
            "scene 只能是 machine_room、near_tower、other。"
            "只输出JSON，格式："
            '{"scene":"other","confidence":0.0,"reason":"简短原因"}'
            f"\nYOLO场景标签：{list(signature)}"
        )
        try:
            parsed = self._generate_json(frame, prompt)
            return (
                _normalize_scene(parsed.get("scene")),
                _confidence(parsed.get("confidence")),
                str(parsed.get("reason", "")),
            )
        except Exception as exc:
            logger.exception("Qwen scene analysis failed, using rule fallback: %s", exc)
            fallback = self._fallback_analysis(detections)
            return fallback.scene, fallback.scene_confidence, "rule_fallback"

    def analyze_tracks(
        self,
        frame: Any,
        detections: list[Detection],
        trajectory_summary: dict[str, Any],
    ) -> TrackAnalysis:
        if not self.ready:
            fallback = self._fallback_analysis(detections)
            return TrackAnalysis(
                height_work=fallback.height_work,
                height_work_confidence=fallback.height_work_confidence,
                briefing=fallback.briefing,
                briefing_confidence=fallback.briefing_confidence,
                reason="rule_fallback",
            )
        prompt = (
            "你是工地施工安全行为审核模型。请结合当前图片、人物轨迹和pose数据判断："
            "1) height_work 是否存在登高作业；"
            "2) briefing 是否存在一人对多人任务交底或安全事项强调。"
            "只输出JSON，格式："
            '{"height_work":false,"height_work_confidence":0.0,'
            '"briefing":false,"briefing_confidence":0.0,"reason":"简短原因"}'
            f"\n轨迹和pose数据：{trajectory_summary}"
            f"\nYOLO检测标签：{_format_detections(detections)}"
        )
        try:
            parsed = self._generate_json(frame, prompt)
            return TrackAnalysis(
                height_work=_bool(parsed.get("height_work")),
                height_work_confidence=_confidence(parsed.get("height_work_confidence")),
                briefing=_bool(parsed.get("briefing")),
                briefing_confidence=_confidence(parsed.get("briefing_confidence")),
                reason=str(parsed.get("reason", "")),
            )
        except Exception as exc:
            logger.exception("Qwen track analysis failed, using rule fallback: %s", exc)
            fallback = self._fallback_analysis(detections)
            return TrackAnalysis(
                height_work=fallback.height_work,
                height_work_confidence=fallback.height_work_confidence,
                briefing=fallback.briefing,
                briefing_confidence=fallback.briefing_confidence,
                reason="rule_fallback",
            )

    def analyze_video_summary(
        self,
        scene_frame: Any,
        summary: dict[str, Any],
        representative_detections: list[Detection],
    ) -> VideoSummaryAnalysis:
        if not self.ready:
            fallback = self._fallback_analysis(representative_detections)
            return VideoSummaryAnalysis(
                scene=fallback.scene,
                scene_confidence=fallback.scene_confidence,
                height_work=fallback.height_work,
                height_work_confidence=fallback.height_work_confidence,
                briefing=fallback.briefing,
                briefing_confidence=fallback.briefing_confidence,
                reason="rule_fallback",
            )

        prompt = (
            "你是工地施工安全视频审核模型。YOLO和pose已经处理完整个视频，"
            "现在给你一张代表场景帧，以及筛选后的检测、告警候选、人物轨迹和pose摘要。"
            "请综合判断："
            "1) scene 只能是 machine_room、near_tower、other；"
            "2) height_work 是否存在登高作业；"
            "3) briefing 是否存在一人对多人任务交底或安全事项强调。"
            "只输出JSON，不要解释。格式："
            '{"scene":"other","scene_confidence":0.0,'
            '"height_work":false,"height_work_confidence":0.0,'
            '"briefing":false,"briefing_confidence":0.0,'
            '"reason":"简短原因"}'
            f"\n筛选后的YOLO/pose摘要：{summary}"
        )
        try:
            parsed = self._generate_json(scene_frame, prompt)
            return VideoSummaryAnalysis(
                scene=_normalize_scene(parsed.get("scene")),
                scene_confidence=_confidence(parsed.get("scene_confidence")),
                height_work=_bool(parsed.get("height_work")),
                height_work_confidence=_confidence(parsed.get("height_work_confidence")),
                briefing=_bool(parsed.get("briefing")),
                briefing_confidence=_confidence(parsed.get("briefing_confidence")),
                reason=str(parsed.get("reason", "")),
            )
        except Exception as exc:
            logger.exception(
                "Qwen video summary analysis failed, using rule fallback: %s", exc
            )
            fallback = self._fallback_analysis(representative_detections)
            return VideoSummaryAnalysis(
                scene=fallback.scene,
                scene_confidence=fallback.scene_confidence,
                height_work=fallback.height_work,
                height_work_confidence=fallback.height_work_confidence,
                briefing=fallback.briefing,
                briefing_confidence=fallback.briefing_confidence,
                reason="rule_fallback",
            )

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
        try:
            parsed = self._generate_json_from_image(image, prompt)
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

    def _generate_json(self, frame: Any, prompt: str) -> dict[str, Any]:
        if self.remote_client is not None and self.remote_client.ready:
            return self.remote_client.generate_json(frame, prompt)
        image = _frame_to_pil(frame)
        return self._generate_json_from_image(image, prompt)

    def _generate_json_from_image(self, image: Any, prompt: str) -> dict[str, Any]:
        if self.remote_client is not None and self.remote_client.ready:
            return self.remote_client.generate_json_from_image(image, prompt)
        assert self.processor is not None
        assert self.model is not None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
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
        logger.debug("Qwen-VL output: %s", output)
        return _parse_json(output)

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


class RemoteMultimodalClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.url = settings.multimodal_api_url
        self.model = settings.multimodal_model or settings.qwen_model
        self.format = settings.multimodal_api_format.lower()
        if not self.url:
            logger.warning(
                "multimodal provider is remote but SUPERVISOR_MULTIMODAL_API_URL is empty; "
                "using rule fallback"
            )
        else:
            logger.info(
                "using remote multimodal service url=%s format=%s model=%s",
                self.url,
                self.format,
                self.model,
            )

    @property
    def ready(self) -> bool:
        return bool(self.url)

    def generate_json(self, frame: Any, prompt: str) -> dict[str, Any]:
        image_data_url = _frame_to_jpeg_data_url(frame)
        return self._post(prompt, image_data_url)

    def generate_json_from_image(self, image: Any, prompt: str) -> dict[str, Any]:
        image_data_url = _pil_to_jpeg_data_url(image)
        return self._post(prompt, image_data_url)

    def _post(self, prompt: str, image_data_url: str) -> dict[str, Any]:
        assert self.url is not None
        headers = {"Content-Type": "application/json"}
        if self.settings.multimodal_api_key:
            headers["Authorization"] = f"Bearer {self.settings.multimodal_api_key}"

        payload = (
            self._openai_compatible_payload(prompt, image_data_url)
            if self.format == "openai_compatible"
            else self._custom_payload(prompt, image_data_url)
        )
        response = httpx.post(
            self.url,
            json=payload,
            headers=headers,
            timeout=self.settings.multimodal_api_timeout,
        )
        response.raise_for_status()
        data = response.json()
        logger.debug("remote multimodal response: %s", data)
        return _extract_json_payload(data)

    def _custom_payload(self, prompt: str, image_data_url: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt": prompt,
            "image": image_data_url,
            "image_base64": image_data_url.split(",", 1)[1],
            "image_mime_type": "image/jpeg",
            "response_format": "json",
        }

    def _openai_compatible_payload(
        self, prompt: str, image_data_url: str
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                    ],
                }
            ],
            "temperature": 0,
        }


def _frame_to_pil(frame: Any) -> Any:
    from PIL import Image

    try:
        import cv2

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return Image.fromarray(frame)


def _frame_to_jpeg_data_url(frame: Any) -> str:
    try:
        import cv2

        ok, buffer = cv2.imencode(".jpg", frame)
        if ok:
            encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        pass
    return _pil_to_jpeg_data_url(_frame_to_pil(frame))


def _pil_to_jpeg_data_url(image: Any) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


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


def _extract_json_payload(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        for key in ("scene", "height_work", "briefing"):
            if key in data:
                return data
        for key in ("json", "result", "output", "content", "text"):
            value = data.get(key)
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                return _parse_json(value)
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return _parse_json(content)
    if isinstance(data, str):
        return _parse_json(data)
    raise ValueError(f"cannot parse multimodal response as JSON: {data}")


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
