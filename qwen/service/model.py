from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from service.config import Settings
from service.openai_compat import parse_model_json_output
from service.prompts import JSON_REPAIR_PROMPT, normalize_prompt

logger = logging.getLogger(__name__)


class QwenVLModel:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.processor: Any | None = None
        self.model: Any | None = None
        self.model_path: str | None = None

    @property
    def ready(self) -> bool:
        return self.settings.mock_mode or (self.processor is not None and self.model is not None)

    def load(self) -> None:
        if self.settings.mock_mode:
            logger.warning("QWEN_MOCK_MODE=true; service will return deterministic mock JSON")
            return
        model_path = self._resolve_model_path()
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                cache_dir=self.settings.hf_cache_dir,
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                device_map=self.settings.device_map,
                torch_dtype="auto",
                trust_remote_code=True,
                cache_dir=self.settings.hf_cache_dir,
            )
            self.model_path = str(model_path)
            logger.info("loaded Qwen-VL model from %s", model_path)
        except Exception as exc:
            logger.exception("failed to load Qwen-VL model: %s", exc)
            raise

    def generate_json(self, image: Image.Image, prompt: str) -> dict[str, Any]:
        if self.settings.mock_mode:
            return _mock_response(prompt)
        if not self.ready:
            self.load()
        try:
            return self._generate_once(image, normalize_prompt(prompt))
        except Exception as first_exc:
            logger.warning("first JSON generation failed, retrying once: %s", first_exc)
            return self._generate_once(image, normalize_prompt(f"{prompt}\n{JSON_REPAIR_PROMPT}"))

    def _generate_once(self, image: Image.Image, prompt: str) -> dict[str, Any]:
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
            messages,
            tokenize=False,
            add_generation_prompt=True,
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
            max_new_tokens=self.settings.max_new_tokens,
        )
        input_length = inputs["input_ids"].shape[-1]
        output_ids = generated_ids[:, input_length:]
        output = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return parse_model_json_output(output)

    def _resolve_model_path(self) -> str:
        if self.settings.model_path and Path(self.settings.model_path).exists():
            return self.settings.model_path
        if self.settings.use_modelscope:
            from modelscope import snapshot_download

            return snapshot_download(
                model_id=self.settings.model_name,
                cache_dir=self.settings.modelscope_cache_dir,
            )
        return self.settings.model_name


def _mock_response(prompt: str) -> dict[str, Any]:
    if "height_work" in prompt:
        return {"confirmed_candidates": []}
    if "briefing" in prompt:
        return {"confirmed_candidates": []}
    if "scene" in prompt:
        return {"scene": "other", "scene_confidence": 0.5, "reason": "mock response"}
    return {"result": "mock", "reason": "QWEN_MOCK_MODE=true"}
