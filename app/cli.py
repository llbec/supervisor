from __future__ import annotations

import argparse

from app.config import get_settings
from app.db import SessionLocal, init_db
from app.logging_config import configure_logging
from app.models import VideoJob
from app.processor import VideoProcessor


def main() -> None:
    configure_logging(get_settings().log_level)
    parser = argparse.ArgumentParser(description="Process one construction-site video.")
    parser.add_argument("source", help="Local video path or stream URL.")
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--source-type", choices=["offline", "stream"], default="offline")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        job = VideoJob(
            source=args.source,
            source_type=args.source_type,
            camera_id=args.camera_id,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        VideoProcessor(get_settings()).process_job(db, job.id)
        db.refresh(job)
        print(f"job_id={job.id} status={job.status}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
