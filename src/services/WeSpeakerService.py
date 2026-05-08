from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import wespeaker


@dataclass
class WeSpeakerService:
    """Small wrapper around the official WeSpeaker model API."""

    language: str = "english"
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Load the model once and keep it in memory for reuse."""
        self.model = wespeaker.load_model(self.language)
        self.model.set_device(self.device)

    def get_embedding(self, audio_path: str | Path) -> np.ndarray:
        """Extract a speaker embedding from an audio file."""
        embedding = self.model.extract_embedding(str(audio_path))
        return np.asarray(embedding)

    def get_similarity(self, audio_path_1: str | Path, audio_path_2: str | Path) -> float:
        """Compute similarity between two audio files."""
        similarity = self.model.compute_similarity(str(audio_path_1), str(audio_path_2))
        return float(similarity)

    def register_speaker(self, speaker_name: str, audio_path: str | Path) -> None:
        """Register one reference recording for a speaker."""
        self.model.register(speaker_name, str(audio_path))

    def recognize_speaker(self, audio_path: str | Path) -> Any:
        """Recognize a speaker using the registered speaker set."""
        return self.model.recognize(str(audio_path))