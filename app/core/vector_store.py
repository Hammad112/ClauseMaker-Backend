"""Qdrant vector store wrapper.

Uses qdrant-client in two modes:
- Production: connects to Qdrant Cloud via QDRANT_URL + QDRANT_API_KEY.
- Test: uses qdrant-client's `:memory:` mode (real Qdrant, just in-process, no network).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.core.config import get_settings


@dataclass
class SearchResult:
    id: str
    score: float
    payload: dict[str, Any]


class VectorStore:
    def __init__(self, collection: str, dim: int = 384) -> None:
        self.collection = collection
        self.dim = dim
        settings = get_settings()
        if settings.is_test_mode or not settings.qdrant_url:
            logger.info("Qdrant: using in-memory mode")
            self.client = QdrantClient(":memory:")
        else:
            logger.info(f"Qdrant: connecting to {settings.qdrant_url}")
            self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]

        def _create() -> None:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {self.collection}")

        if self.collection not in existing:
            _create()
        else:
            # Collection already exists. Verify it matches the expected schema
            # (unnamed 384-d vectors). If not — typical when a user pre-creates
            # the collection via the Qdrant Cloud UI which defaults to named
            # vectors — drop and recreate. Safe because the frameworks corpus
            # is small and re-upserted on boot via `ensure_frameworks_loaded`.
            try:
                info = self.client.get_collection(self.collection)
                params = info.config.params.vectors  # type: ignore[union-attr]
                schema_ok = (
                    hasattr(params, "size")
                    and getattr(params, "size", None) == self.dim
                )
                if not schema_ok:
                    logger.warning(
                        f"Qdrant collection '{self.collection}' has an incompatible "
                        f"vector schema ({params!r}). Recreating with unnamed "
                        f"{self.dim}-d cosine vectors."
                    )
                    self.client.delete_collection(self.collection)
                    _create()
            except Exception as e:
                logger.warning(
                    f"Could not introspect collection '{self.collection}', "
                    f"recreating: {e}"
                )
                try:
                    self.client.delete_collection(self.collection)
                except Exception:
                    pass
                _create()

        # Qdrant Cloud requires a payload index for filtered count/search.
        # Idempotent — safe to call every boot. The in-memory client is more
        # lenient but the call is still cheap there.
        try:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="framework_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as e:  # already exists is fine
            logger.debug(f"framework_id payload index already present: {e}")

    def upsert(self, points: list[tuple[str, np.ndarray, dict[str, Any]]]) -> None:
        """points: list of (id, vector, payload)."""
        structs = [
            PointStruct(id=pid, vector=vec.tolist(), payload=payload)
            for pid, vec, payload in points
        ]
        self.client.upsert(collection_name=self.collection, points=structs)

    def search(
        self,
        query_vector: np.ndarray,
        framework_id: str,
        top_k: int = 15,
    ) -> list[SearchResult]:
        """Search filtered by framework_id."""
        flt = Filter(
            must=[FieldCondition(key="framework_id", match=MatchValue(value=framework_id))]
        )
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            query_filter=flt,
            limit=top_k,
            with_payload=True,
        ).points
        return [
            SearchResult(id=str(r.id), score=float(r.score), payload=dict(r.payload or {}))
            for r in results
        ]

    def count(self, framework_id: str | None = None) -> int:
        flt = None
        if framework_id:
            flt = Filter(must=[FieldCondition(key="framework_id", match=MatchValue(value=framework_id))])
        return self.client.count(collection_name=self.collection, count_filter=flt).count

    def article_ids_for_framework(self, framework_id: str) -> set[str]:
        """Return the set of all article_ids indexed for a framework. Used by the citation validator."""
        flt = Filter(must=[FieldCondition(key="framework_id", match=MatchValue(value=framework_id))])
        ids: set[str] = set()
        offset = None
        while True:
            batch, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=flt,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in batch:
                payload = point.payload or {}
                aid = payload.get("article_id")
                if aid:
                    ids.add(str(aid))
            if offset is None:
                break
        return ids


_vector_store_singleton: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store_singleton
    if _vector_store_singleton is None:
        settings = get_settings()
        _vector_store_singleton = VectorStore(collection=settings.qdrant_collection)
    return _vector_store_singleton


def reset_vector_store_singleton() -> None:
    """For tests."""
    global _vector_store_singleton
    _vector_store_singleton = None
