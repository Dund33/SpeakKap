from __future__ import annotations

from typing import Iterable

import numpy as np


class MathUtils:

    @staticmethod
    def cosine_distance(
        a: np.ndarray,
        b: np.ndarray
    ) -> float:

        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)

        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)

        if a_norm == 0 or b_norm == 0:
            raise ValueError("Zero vector")

        similarity = np.dot(a, b) / (
            a_norm * b_norm
        )

        return float(1.0 - similarity)

    @staticmethod
    def get_medoid(
        vectors: Iterable[np.ndarray]
    ) -> np.ndarray:

        vectors = [
            np.asarray(v, dtype=np.float32)
            for v in vectors
        ]

        if len(vectors) == 0:
            raise ValueError(
                "Vector list cannot be empty"
            )

        if len(vectors) == 1:
            return vectors[0]

        best_index = 0
        best_distance = float("inf")

        for i in range(len(vectors)):

            total_distance = 0.0

            for j in range(len(vectors)):

                if i == j:
                    continue

                total_distance += (
                    MathUtils.cosine_distance(
                        vectors[i],
                        vectors[j]
                    )
                )

            if total_distance < best_distance:
                best_distance = total_distance
                best_index = i

        return vectors[best_index]