"""Citation validator.

The single most important guardrail in the system. Every Article ID cited by the LLM
must exist in the indexed corpus. Hallucinated citations are silently dropped and
the affected mapping is flagged as low-confidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from loguru import logger

from app.core.vector_store import get_vector_store


@dataclass
class ValidationResult:
    valid_count: int
    dropped_article_ids: list[str]


@lru_cache(maxsize=8)
def _valid_article_ids(framework_id: str) -> frozenset[str]:
    """Cached set of valid article IDs for a framework."""
    store = get_vector_store()
    return frozenset(store.article_ids_for_framework(framework_id))


def refresh_cache() -> None:
    """Clear the cache — call after re-indexing a framework."""
    _valid_article_ids.cache_clear()


def is_valid_citation(framework_id: str, article_id: str) -> bool:
    return article_id in _valid_article_ids(framework_id)


def validate_mappings(framework_id: str, article_ids: list[str]) -> ValidationResult:
    valid_ids = _valid_article_ids(framework_id)
    if not valid_ids:
        logger.error(f"No valid article IDs found for framework {framework_id}")
        return ValidationResult(valid_count=0, dropped_article_ids=article_ids)

    dropped = []
    valid = 0
    for aid in article_ids:
        if aid in valid_ids:
            valid += 1
        else:
            dropped.append(aid)
            logger.warning(f"Citation validator dropped hallucinated article_id={aid}")

    return ValidationResult(valid_count=valid, dropped_article_ids=dropped)
