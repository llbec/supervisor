from __future__ import annotations

import base64
import json
import re
import time
import uuid
from io import BytesIO
from typing import Any

from fastapi import HTTPException
from PIL import Image

from service.schemas import ChatCompletionRequest, ParsedRequest


def parse_chat_completion_request(request: ChatCompletionRequest) -> ParsedRequest:
    text_parts: list[str] = []
    image_url: str | None = None
    for message in request.messages:
        if isinstance(message.content, str):
            text_parts.append(message.content)
            continue
        for part in message.content:
            if part.type == "text" and part.text:
                text_parts.append(part.text)
            if part.type == "image_url" and part.image_url and not image_url:
                image_url = part.image_url.url
    if not image_url:
        raise HTTPException(status_code=400, detail="missing image_url in messages")
    prompt = "\n".join(text_parts).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="missing text prompt in messages")
    return ParsedRequest(prompt=prompt, image_url=image_url, task=_extract_task(prompt))


def image_from_url(image_url: str) -> Image.Image:
    if not image_url.startswith("data:image/"):
        raise HTTPException(
            status_code=400,
            detail="only data:image/*;base64 image_url is supported in the first version",
        )
    try:
        _, encoded = image_url.split(",", 1)
        data = base64.b64decode(encoded)
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid base64 image: {exc}") from exc


def build_chat_completion_response(model: str, content: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(content, ensure_ascii=False),
                },
                "finish_reason": "stop",
            }
        ],
    }


def parse_model_json_output(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"model output is not JSON: {text}")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError(f"model JSON output is not an object: {text}")
    return value


def _extract_task(prompt: str) -> str | None:
    match = re.search(r"任务类型[:：]\s*([a-zA-Z_]+)", prompt)
    return match.group(1) if match else None
