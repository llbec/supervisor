from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SUPERVISOR_")

    database_url: str = "sqlite:///./data/supervisor.db"
    frame_sample_interval: int = Field(default=15, ge=1)
    snapshot_dir: str = "data/snapshots"
    log_level: str = "INFO"
    alert_webhook_url: str | None = None

    yolo_seg_model: str = "weights/yoloe-26l-seg.pt"
    yolo_pose_model: str = "weights/yolo26n-pose.pt"
    detection_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    tracker_backend: str = "bytetrack"
    tracker_iou_threshold: float = Field(default=0.25, ge=0.0, le=1.0)

    ppe_required_hits: int = Field(default=2, ge=1)
    ppe_missing_tolerance: int = Field(default=2, ge=0)
    ppe_edge_margin_ratio: float = Field(default=0.03, ge=0.0, le=0.2)

    trajectory_window_ms: int = Field(default=30_000, ge=5_000)
    min_candidate_duration_ms: int = Field(default=2_000, ge=0)

    multimodal_provider: str = "remote"
    multimodal_api_url: str | None = None
    multimodal_api_key: str | None = None
    multimodal_api_format: str = "openai_compatible"
    multimodal_api_timeout: float = Field(default=60.0, ge=1.0)
    multimodal_model: str = "Qwen/Qwen3-VL-8B-Instruct"

    @property
    def sqlite_path(self) -> Path | None:
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            return None
        return Path(self.database_url.removeprefix(prefix))


@lru_cache
def get_settings() -> Settings:
    return Settings()
