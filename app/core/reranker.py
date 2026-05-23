"""Reranker provider.

Production: BAAI/bge-reranker-base via FlagEmbedding.
Test mode: token-overlap heuristic that approximates cross-encoder behavior.
"""
from __future__ import annotations

import re
from typing import Protocol

from loguru import logger

from app.core.config import get_settings


class Reranker(Protocol):
    def score(self, query: str, candidates: list[str]) -> list[float]: ...


class _OverlapReranker:
    """Test-mode reranker.

    Returns a score in roughly [-1, 1] based on Jaccard similarity of token sets
    plus a small boost when candidate contains rare query tokens. Designed to rank
    candidates similarly to a real cross-encoder for our test fixtures.
    """

    def __init__(self) -> None:
        logger.info("OverlapReranker initialized (test mode)")

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t for t in re.findall(r"[a-z]+", text.lower()) if len(t) >= 4}

    def score(self, query: str, candidates: list[str]) -> list[float]:
        q = self._tokens(query)
        if not q:
            return [0.0] * len(candidates)
        scores = []
        for c in candidates:
            ct = self._tokens(c)
            if not ct:
                scores.append(-1.0)
                continue
            inter = q & ct
            union = q | ct
            jaccard = len(inter) / len(union) if union else 0.0
            # boost for rare overlap (signals strong topical match)
            rare_boost = 0.2 if any(t in ct for t in q if len(t) > 7) else 0.0
            scores.append(min(1.0, jaccard * 2.0 + rare_boost - 0.3))
        return scores


class _BGEReranker:
    """Production reranker. Lazy-loaded."""

    def __init__(self) -> None:
        self._model = None

    def _load(self):
        if self._model is None:
            from FlagEmbedding import FlagReranker  # type: ignore
            logger.info("Loading BAAI/bge-reranker-base...")
            self._model = FlagReranker("BAAI/bge-reranker-base", use_fp16=True)
            logger.info("Reranker loaded")

    def score(self, query: str, candidates: list[str]) -> list[float]:
        self._load()
        pairs = [[query, c] for c in candidates]
        # FlagReranker returns raw logits; sigmoid for [0,1]-ish, then center
        raw = self._model.compute_score(pairs, normalize=True)  # type: ignore
        if isinstance(raw, float):
            raw = [raw]
        return [float(s) for s in raw]


_reranker_singleton: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker_singleton
    if _reranker_singleton is None:
        settings = get_settings()
        if settings.is_test_mode or not settings.use_real_reranker:
            _reranker_singleton = _OverlapReranker()
        else:
            _reranker_singleton = _BGEReranker()
    return _reranker_singleton
