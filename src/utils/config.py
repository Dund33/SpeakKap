from __future__ import annotations

import os


class Config:
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    WESPEAKER_DEVICE = os.getenv("WESPEAKER_DEVICE", "cpu")

    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "256"))

    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")

    THRESHOLD = float(os.getenv("THRESHOLD", "0.15"))
