"""Clause mapper — the orchestration layer for the mapping pipeline.

Steps per clause:
  1. embed
  2. retrieve top-15 candidates from Qdrant (filtered by framework)
  3. rerank with cross-encoder, keep top-5
  4. ask LLM to score and classify each candidate
  5. validate citations (drop hallucinated article_ids)
  6. fuse confidence scores
  7. return list of ClauseMappingResult
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from app.core.embeddings import get_embedder
from app.core.llm import CandidateArticle, LLMMappingResult, get_llm
from app.core.reranker import get_reranker
from app.core.vector_store import get_vector_store, SearchResult
from app.services import citation_validator
from app.services.confidence_scorer import fuse_confidence, normalize_min_max


@dataclass
class ClauseMappingResult:
    article_id: str
    article_title: str
    article_text: str
    framework_id: str
    classification: str
    confidence: float
    retrieval_score: float
    reranker_score: float
    llm_self_score: float
    reasoning: str
    gap_remediation: str | None
    source_url: str | None


def _candidates_from_retrieval(results: list[SearchResult]) -> list[CandidateArticle]:
    out = []
    for r in results:
        p = r.payload
        out.append(CandidateArticle(
            article_id=str(p.get("article_id", "")),
            article_title=str(p.get("article_title", "")),
            article_text=str(p.get("full_text", "")),
        ))
    return out


def map_clauses(
    clauses: list[tuple[str, str]],  # list of (heading_path, text)
    framework_id: str,
    top_k_retrieve: int = 15,
    top_k_rerank: int = 5,
) -> list[list[ClauseMappingResult]]:
    """Map a batch of clauses. Returns a list of mapping-result lists (one per clause)."""
    if not clauses:
        return []

    embedder = get_embedder()
    reranker = get_reranker()
    store = get_vector_store()
    llm = get_llm()

    texts = [text for _, text in clauses]
    embeddings = embedder.encode(texts)

    all_results: list[list[ClauseMappingResult]] = []

    for i, (heading_path, clause_text) in enumerate(clauses):
        # 1. Retrieve
        retrieval_results = store.search(
            query_vector=embeddings[i],
            framework_id=framework_id,
            top_k=top_k_retrieve,
        )

        if not retrieval_results:
            all_results.append([])
            continue

        # 2. Rerank
        candidate_texts = [r.payload.get("full_text", "") for r in retrieval_results]
        reranker_scores = reranker.score(clause_text, candidate_texts)

        # Combine retrieval results with reranker scores, take top-k by reranker
        scored_idx = sorted(range(len(retrieval_results)), key=lambda k: -reranker_scores[k])[:top_k_rerank]
        top_retrieval = [retrieval_results[k] for k in scored_idx]
        top_reranker = [reranker_scores[k] for k in scored_idx]

        # 3. LLM mapping
        candidates = _candidates_from_retrieval(top_retrieval)
        llm_results = llm.map_clause(
            clause_text=clause_text,
            heading_path=heading_path,
            candidates=candidates,
        )

        # If LLM returned nothing, fall back to retrieval-only with low confidence
        if not llm_results:
            logger.warning(f"LLM returned no results for clause '{heading_path}'")
            llm_results = [
                LLMMappingResult(
                    candidate_index=j,
                    article_id=c.article_id,
                    relevance_score=top_retrieval[j].score * 100,
                    classification="Partial",
                    reasoning="(LLM unavailable — fallback to retrieval similarity)",
                    gap_remediation=None,
                    confidence=40.0,
                )
                for j, c in enumerate(candidates)
            ]

        # 4. Validate citations and prepare normalization inputs
        cited_ids = [lr.article_id for lr in llm_results]
        validation = citation_validator.validate_mappings(framework_id, cited_ids)
        dropped_set = set(validation.dropped_article_ids)

        retrieval_scores_for_llm = []
        reranker_scores_for_llm = []
        # Build lookup from article_id to retrieval/reranker score
        lookup = {
            c.article_id: (top_retrieval[j].score, top_reranker[j])
            for j, c in enumerate(candidates)
        }
        for lr in llm_results:
            ret, rer = lookup.get(lr.article_id, (0.0, 0.0))
            retrieval_scores_for_llm.append(ret)
            reranker_scores_for_llm.append(rer)

        # Normalize within this clause's candidate set
        norm_retrieval = normalize_min_max(retrieval_scores_for_llm)
        norm_reranker = normalize_min_max(reranker_scores_for_llm)

        # 5. Fuse confidence + build final results
        clause_results: list[ClauseMappingResult] = []
        for j, lr in enumerate(llm_results):
            if lr.article_id in dropped_set:
                logger.warning(f"Skipping hallucinated citation: {lr.article_id}")
                continue
            cand = next((c for c in candidates if c.article_id == lr.article_id), None)
            if cand is None:
                continue
            source_url = next(
                (r.payload.get("source_url") for r in top_retrieval if r.payload.get("article_id") == lr.article_id),
                None,
            )
            confidence = fuse_confidence(
                retrieval_normalized=norm_retrieval[j],
                reranker_normalized=norm_reranker[j],
                llm_self_score=lr.confidence,
                citation_validated=True,
            )
            clause_results.append(ClauseMappingResult(
                article_id=lr.article_id,
                article_title=cand.article_title,
                article_text=cand.article_text,
                framework_id=framework_id,
                classification=lr.classification,
                confidence=confidence,
                retrieval_score=round(retrieval_scores_for_llm[j], 4),
                reranker_score=round(reranker_scores_for_llm[j], 4),
                llm_self_score=round(lr.confidence, 1),
                reasoning=lr.reasoning,
                gap_remediation=lr.gap_remediation,
                source_url=source_url,
            ))

        # Sort by confidence descending
        clause_results.sort(key=lambda r: -r.confidence)
        all_results.append(clause_results)

    return all_results
