from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateAnalysis:
    confirmed_candidates: list[dict[str, Any]]
    raw: dict[str, Any]


class MultimodalClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def ready(self) -> bool:
        return bool(self.settings.multimodal_api_url)

    def analyze_scene(self, image, summary: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "任务类型：scene。请根据图片和结构化摘要判断施工场景。"
            "scene 只能是 machine_room、near_tower、other。只输出合法JSON。"
            f"\n结构化摘要：{json.dumps(summary, ensure_ascii=False)}"
        )
        return self._call_json(image, prompt, fallback={"scene": "other", "scene_confidence": 0.0, "reason": "multimodal_unavailable"})

    def analyze_height_work(self, image, candidates: list[dict[str, Any]]) -> CandidateAnalysis:
        if not candidates:
            return CandidateAnalysis([], {"reason": "no_height_work_candidates"})
        prompt = (
            "任务类型：height_work。你只能判断输入候选片段是否存在登高作业。"
            "如果没有人员靠近高处结构、没有垂直移动、没有攀爬或高处作业pose，不得判定为登高。"
            "返回 confirmed_candidates 数组，只输出合法JSON。"
            f"\n候选片段：{json.dumps(candidates, ensure_ascii=False)}"
        )
        raw = self._call_json(image, prompt, fallback={"confirmed_candidates": []})
        return CandidateAnalysis(raw.get("confirmed_candidates", []), raw)

    def analyze_briefing(self, image, candidates: list[dict[str, Any]]) -> CandidateAnalysis:
        if not candidates:
            return CandidateAnalysis([], {"reason": "no_briefing_candidates"})
        prompt = (
            "任务类型：briefing。你只能判断输入候选片段是否存在交底行为。"
            "多人场景关注一人讲解多人听取；两人场景必须结合文档类物品、持续停留和签字/确认动作。"
            "普通交谈、短暂同框、普通施工动作不得判定为交底。返回 confirmed_candidates 数组，只输出合法JSON。"
            f"\n候选片段：{json.dumps(candidates, ensure_ascii=False)}"
        )
        raw = self._call_json(image, prompt, fallback={"confirmed_candidates": []})
        return CandidateAnalysis(raw.get("confirmed_candidates", []), raw)

    def _call_json(self, image, prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
        if not self.ready:
            return fallback
        payload = self._openai_payload(image, prompt)
        headers = {"Content-Type": "application/json"}
        if self.settings.multimodal_api_key:
            headers["Authorization"] = f"Bearer {self.settings.multimodal_api_key}"
        try:
            response = httpx.post(
                self.settings.multimodal_api_url,
                json=payload,
                headers=headers,
                timeout=self.settings.multimodal_api_timeout,
            )
            response.raise_for_status()
            return _extract_json(response.json())
        except Exception as exc:
            logger.exception("multimodal API failed; using fallback: %s", exc)
            return fallback

    def _openai_payload(self, image, prompt: str) -> dict[str, Any]:
        return {
            "model": self.settings.multimodal_model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是工地施工安全视频审核模型。只输出合法JSON，不要输出解释性文本。",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _frame_to_data_url(image)},
                        },
                    ],
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }


def _frame_to_data_url(frame) -> str:
    import cv2

    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("failed to encode frame as jpeg")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_json(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        if any(key in data for key in ("scene", "confirmed_candidates")):
            return data
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            content = choices[0].get("message", {}).get("content")
            if isinstance(content, dict):
                return content
            if isinstance(content, str):
                return _parse_json(content)
        for key in ("json", "result", "output", "content", "text"):
            value = data.get(key)
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                return _parse_json(value)
    if isinstance(data, str):
        return _parse_json(data)
    raise ValueError(f"cannot parse multimodal response: {data}")


def _parse_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))
