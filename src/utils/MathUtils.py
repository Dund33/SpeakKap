from __future__ import annotations

from typing import Iterable

import numpy as np


class MathUtils:
    """Utility methods for computing medoids from NumPy vectors."""

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine distance between two vectors.

        Returns:
            float: cosine distance in range [0, 2]
        """

        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)

        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)

        if a_norm == 0 or b_norm == 0:
            raise ValueError("Vectors must not be zero vectors")

        similarity = np.dot(a, b) / (a_norm * b_norm)

        return 1.0 - similarity

    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Compute Euclidean distance between two vectors."""

        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)

        return float(np.linalg.norm(a - b))

    @staticmethod
    def get_medoid(
        vectors: Iterable[np.ndarray],
        metric: str = "cosine",
    ) -> np.ndarray:
        """Compute the medoid vector from a collection of vectors.

        The medoid is the real vector from the input set that minimizes
        the total distance to all other vectors.

        Args:
            vectors:
                Iterable containing NumPy arrays.

            metric:
                Distance metric to use.
                Supported values:
                    - "cosine"
                    - "euclidean"

        Returns:
            np.ndarray: medoid vector
        """

        vectors = [
            np.asarray(v, dtype=np.float32)
            for v in vectors
        ]

        if len(vectors) == 0:
            raise ValueError("Vector list cannot be empty")

        if len(vectors) == 1:
            return vectors[0]

        vector_shapes = {
            v.shape for v in vectors
        }

        if len(vector_shapes) != 1:
            raise ValueError(
                "All vectors must have the same shape"
            )

        if metric == "cosine":
            distance_fn = MathUtils.cosine_distance

        elif metric == "euclidean":
            distance_fn = MathUtils.euclidean_distance

        else:
            raise ValueError(
                "Unsupported metric. Use 'cosine' or 'euclidean'"
            )

        num_vectors = len(vectors)

        distance_matrix = np.zeros(
            (num_vectors, num_vectors),
            dtype=np.float32
        )

        # Compute pairwise distances
        for i in range(num_vectors):
            for j in range(i + 1, num_vectors):

                dist = distance_fn(
                    vectors[i],
                    vectors[j]
                )

                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        # Sum distances for each vector
        total_distances = np.sum(
            distance_matrix,
            axis=1
        )

        # Index of the medoid
        medoid_index = int(
            np.argmin(total_distances)
        )

        return vectors[medoid_index]