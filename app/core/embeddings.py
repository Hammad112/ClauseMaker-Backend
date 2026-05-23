"""Embeddings provider.

Production: BAAI/bge-small-en-v1.5 via sentence-transformers (384 dims).
Test mode: deterministic hash-based embeddings (384 dims) — same input always returns same vector,
similar inputs return similar vectors. Good enough for end-to-end pipeline tests.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import numpy as np
from loguru import logger

from app.core.config import get_settings


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class _HashEmbedder:
    """Hash-based bag-of-tokens embedder for tests.

    Produces a 384-dim sparse-ish vector where each token contributes via a hash
    to a random subset of dimensions. Cosine similarity between texts with overlapping
    vocabulary is meaningfully higher than between unrelated texts.
    """

    dim = 384

    def __init__(self) -> None:
        logger.info("HashEmbedder initialized (test mode)")

    @staticmethod
    def _tokens(text: str) -> list[str]:
        # lowercase, alpha tokens, length >= 3
        return [t for t in re.findall(r"[a-z]+", text.lower()) if len(t) >= 3]

    def _vector_for(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in self._tokens(text):
            h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
            # spread token across 8 dimensions with deterministic sign
            for k in range(8):
                idx = (h >> (k * 8)) & 0xFFFF
                sign = 1.0 if ((h >> (k * 4)) & 1) == 1 else -1.0
                v[idx % self.dim] += sign
        # L2-normalize
        norm = math.sqrt(float((v * v).sum()))
        if norm > 0:
            v /= norm
        return v

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vector_for(t) for t in texts])


class _BGEEmbedder:
    """Production BGE-small embedder. Lazy-loaded on first encode call."""

    dim = 384

    def __init__(self) -> None:
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            logger.info("Loading BAAI/bge-small-en-v1.5 model...")
            self._model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            logger.info("BGE model loaded")

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        # BGE recommends an instruction prefix for retrieval queries
        prefixed = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        return self._model.encode(prefixed, batch_size=32, normalize_embeddings=True)  # type: ignore


_embedder_singleton: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder_singleton
    if _embedder_singleton is None:
        settings = get_settings()
        if settings.is_test_mode or not settings.use_real_embedder:
            _embedder_singleton = _HashEmbedder()
        else:
            _embedder_singleton = _BGEEmbedder()
    return _embedder_singleton
