from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SUPERVISOR_")

    database_url: str = "sqlite:///./data/supervisor.db"
    frame_sample_interval: int = Field(default=15, ge=1)
    alert_webhook_url: str | None = None

    yolo_seg_model: str = "weights/yoloe-26l-seg.pt"
    yolo_pose_model: str = "weights/yolo26n-pose.pt"
    qwen_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    mobilenet_model: str = "weights/mobilenetv3.pth"

    detection_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    realtime_alert_labels: set[str] = {"ppe_violation", "smoking", "hot_work"}

    @property
    def sqlite_path(self) -> Path | None:
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            return None
        return Path(self.database_url.removeprefix(prefix))


@lru_cache
def get_settings() -> Settings:
    return Settings()
