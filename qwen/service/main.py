from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, Header, HTTPException

from service.config import Settings, get_settings
from service.model import QwenVLModel
from service.openai_compat import (
    build_chat_completion_response,
    image_from_url,
    parse_chat_completion_request,
)
from service.schemas import ChatCompletionRequest

logger = logging.getLogger(__name__)
app = FastAPI(title="Local Qwen Multimodal Service", version="0.1.0")
model: QwenVLModel | None = None


@app.on_event("startup")
def startup() -> None:
    global model
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    model = QwenVLModel(settings)
    if settings.load_on_startup:
        model.load()


def require_auth(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.api_key:
        return
    expected = f"Bearer {settings.api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None and model.ready,
        "mock_mode": get_settings().mock_mode,
    }


@app.get("/v1/models")
def list_models(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_name,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(require_auth)])
def chat_completions(request: ChatCompletionRequest) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="model is not initialized")
    parsed = parse_chat_completion_request(request)
    image = image_from_url(parsed.image_url)
    result = model.generate_json(image, parsed.prompt)
    return build_chat_completion_response(model=request.model, content=result)
