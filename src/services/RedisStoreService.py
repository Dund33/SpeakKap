from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import bcrypt
import numpy as np
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.exceptions import ResponseError


@dataclass
class SpeakerProfile:
    """In-memory representation of a stored speaker profile."""

    login: str
    password_hash: str
    embedding: np.ndarray


class RedisStoreService:
    """Store speaker profiles in Redis using a hash + vector index.

    Redis key format:
        speaker:{login}

    Stored hash fields:
        - login (TEXT)
        - password_hash (TEXT)
        - embedding (VECTOR, raw FLOAT32 bytes)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        index_name: str = "speaker_idx",
        prefix: str = "speaker:",
        embedding_dim: int = 512,
    ) -> None:
        self.client = redis.Redis.from_url(redis_url, decode_responses=False)
        self.index_name = index_name
        self.prefix = prefix
        self.embedding_dim = embedding_dim

    def _key(self, login: str) -> str:
        return f"{self.prefix}{login}"

    @staticmethod
    def hash_password(password: str) -> str:
        """Create a bcrypt password hash."""
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify a plaintext password against a bcrypt hash."""
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))

    @staticmethod
    def _to_vector_bytes(embedding: np.ndarray) -> bytes:
        """Convert an embedding to Redis vector bytes (FLOAT32)."""
        return np.asarray(embedding, dtype=np.float32).tobytes()

    @staticmethod
    def _from_vector_bytes(payload: bytes) -> np.ndarray:
        """Convert Redis vector bytes back to a NumPy array."""
        return np.frombuffer(payload, dtype=np.float32)

    def ensure_index(self) -> None:
        """Create the vector index if it does not already exist."""
        schema = (
            TextField("login"),
            TextField("password_hash"),
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self.embedding_dim,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        )

        try:
            self.client.ft(self.index_name).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[self.prefix],
                    index_type=IndexType.HASH,
                ),
            )
        except ResponseError as exc:
            # Redis returns an error if the index already exists.
            if "Index already exists" not in str(exc):
                raise

    def save_profile(self, login: str, password_hash: str, embedding: np.ndarray) -> None:
        """Store a profile in Redis."""
        self.client.hset(
            self._key(login),
            mapping={
                b"login": login.encode("utf-8"),
                b"password_hash": password_hash.encode("utf-8"),
                b"embedding": self._to_vector_bytes(embedding),
            },
        )

    def create_profile(self, login: str, password: str, embedding: np.ndarray) -> None:
        """Create a profile from a plaintext password and an embedding."""
        self.save_profile(login, self.hash_password(password), embedding)

    def get_profile(self, login: str) -> Optional[SpeakerProfile]:
        """Load a stored profile from Redis."""
        data = self.client.hgetall(self._key(login))
        if not data:
            return None

        return SpeakerProfile(
            login=data[b"login"].decode("utf-8"),
            password_hash=data[b"password_hash"].decode("utf-8"),
            embedding=self._from_vector_bytes(data[b"embedding"]),
        )

    def get_embedding(self, login: str) -> Optional[np.ndarray]:
        """Return only the stored embedding for a given login."""
        profile = self.get_profile(login)
        return None if profile is None else profile.embedding

    def authenticate(self, login: str, password: str) -> bool:
        """Check whether the provided password matches the stored hash."""
        profile = self.get_profile(login)
        if profile is None:
            return False
        return self.verify_password(password, profile.password_hash)

    def find_similar_speakers(self, embedding: np.ndarray, top_k: int = 5):
        """Search Redis for the nearest stored speaker embeddings."""
        query = f"*=>[KNN {top_k} @embedding $vec AS score]"
        res = self.client.ft(self.index_name).search(
            query,
            query_params={"vec": self._to_vector_bytes(embedding)},
        )
        return res

    def update_embedding(self, login: str, embedding: np.ndarray) -> bool:
        """Update the stored embedding for an existing profile."""
        if not self.client.exists(self._key(login)):
            return False

        self.client.hset(self._key(login), mapping={b"embedding": self._to_vector_bytes(embedding)})
        return True

    def delete_profile(self, login: str) -> bool:
        """Delete a speaker profile."""
        return bool(self.client.delete(self._key(login)))


if __name__ == "__main__":
    service = RedisStoreService(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        embedding_dim=512,
    )

    # Create the index once during startup.
    service.ensure_index()

    sample_embedding = np.random.rand(512).astype(np.float32)

    service.create_profile(
        login="admin",
        password="secret123",
        embedding=sample_embedding,
    )

    profile = service.get_profile("admin")
    print(profile)

    print("Authenticated:", service.authenticate("admin", "secret123"))
