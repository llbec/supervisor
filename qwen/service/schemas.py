from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ImageUrl(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageUrl | None = None


class ChatMessage(BaseModel):
    role: str
    content: str | list[ContentPart]


class ResponseFormat(BaseModel):
    type: str = "json_object"


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 0
    response_format: ResponseFormat | None = None
    max_tokens: int | None = None
    max_new_tokens: int | None = None


class ParsedRequest(BaseModel):
    prompt: str
    image_url: str
    task: str | None = None


class ErrorDetail(BaseModel):
    message: str
    type: str
    code: str | None = None


def error_response(message: str, error_type: str, code: str | None = None) -> dict[str, Any]:
    return {"error": ErrorDetail(message=message, type=error_type, code=code).model_dump()}
