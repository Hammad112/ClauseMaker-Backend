"""Confidence scorer.

Fuses three signals into a final 0-100 confidence:
- retrieval similarity (0-1) - 30% weight
- reranker score (0-1) - 30% weight
- LLM self-assessment (0-100) - 40% weight
"""
from __future__ import annotations


def normalize_min_max(values: list[float]) -> list[float]:
    """Min-max normalize a list to [0, 1]. Returns all 0.5 if all values equal."""
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mx - mn < 1e-9:
        return [0.5] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def fuse_confidence(
    retrieval_normalized: float,
    reranker_normalized: float,
    llm_self_score: float,
    *,
    citation_validated: bool = True,
) -> float:
    """Return final confidence in 0-100.

    If citation_validated is False (LLM cited a non-existent article), confidence
    is reduced to at most 40.
    """
    base = (
        0.30 * retrieval_normalized
        + 0.30 * reranker_normalized
        + 0.40 * (llm_self_score / 100.0)
    ) * 100.0
    if not citation_validated:
        base = min(base, 40.0)
    return round(max(0.0, min(100.0, base)), 1)


def primary_status_for_clause(mapping_classifications: list[tuple[str, float]]) -> tuple[str, float]:
    """Determine the primary status for a clause given all its mappings.

    Rule:
    - If any mapping is Compliant with confidence >= 60: primary = Compliant
    - Else if any mapping is Partial with confidence >= 50: primary = Partial
    - Else if any mapping is Gap with confidence >= 40: primary = Gap
    - Else: NotApplicable

    Returns (status, confidence_of_that_mapping).
    """
    if not mapping_classifications:
        return "NotApplicable", 0.0

    # Sort by confidence descending within each priority bucket
    compliants = [(c, conf) for c, conf in mapping_classifications if c == "Compliant" and conf >= 60]
    if compliants:
        compliants.sort(key=lambda x: -x[1])
        return compliants[0]

    partials = [(c, conf) for c, conf in mapping_classifications if c == "Partial" and conf >= 50]
    if partials:
        partials.sort(key=lambda x: -x[1])
        return partials[0]

    gaps = [(c, conf) for c, conf in mapping_classifications if c == "Gap" and conf >= 40]
    if gaps:
        gaps.sort(key=lambda x: -x[1])
        return gaps[0]

    return "NotApplicable", max((conf for _, conf in mapping_classifications), default=0.0)
