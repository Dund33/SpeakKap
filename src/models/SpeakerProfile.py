from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpeakerProfile:
    login: str
    password_hash: str
    embedding: np.ndarray
