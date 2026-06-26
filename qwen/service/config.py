from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QWEN_", env_file=".env")

    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    model_path: str = ""
    use_modelscope: bool = True
    modelscope_cache_dir: str = "data/modelscope"
    hf_cache_dir: str = "data/huggingface"
    device_map: str = "auto"
    max_new_tokens: int = Field(default=512, ge=16)
    api_key: str = ""
    mock_mode: bool = False
    load_on_startup: bool = True
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
