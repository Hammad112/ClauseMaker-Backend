"""LLM provider.

Production: Groq Llama 3.3 70B primary, Gemini 2.5 Flash fallback on 429.
Test mode: deterministic rule-based mapper that produces realistic JSON output
using token overlap + simple heuristics. No API calls, fully reproducible.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from loguru import logger

from app.core.config import get_settings


@dataclass
class LLMMappingResult:
    candidate_index: int
    article_id: str
    relevance_score: float  # 0-100
    classification: str  # Compliant|Partial|Gap|NotApplicable
    reasoning: str
    gap_remediation: str | None
    confidence: float  # 0-100 LLM self-confidence


@dataclass
class CandidateArticle:
    article_id: str
    article_title: str
    article_text: str


class LLM(Protocol):
    def map_clause(
        self,
        clause_text: str,
        heading_path: str,
        candidates: list[CandidateArticle],
    ) -> list[LLMMappingResult]: ...


# ----------------- Mock LLM (test mode) -----------------

class _MockLLM:
    """Rule-based LLM stand-in.

    Heuristics for classification:
    - Token overlap with article > 35% AND clause is on-topic for AI/data/privacy => Compliant
    - Overlap 15-35% AND mentions risk/data/policy concepts => Partial
    - Overlap < 15% but topical => Gap
    - Off-topic (e.g. financial/marketing language vs AI Act) => NotApplicable

    Confidence = overlap * 100 with small randomness based on hash.
    """

    REGULATORY_KEYWORDS = {
        "risk", "data", "policy", "governance", "transparency", "oversight",
        "accuracy", "robustness", "discrimination", "bias", "human", "automated",
        "decision", "documentation", "log", "logging", "monitor", "retention",
        "consent", "purpose", "minimization", "training", "model", "system",
        "personal", "subject", "controller", "processor", "right", "fundamental",
        "audit", "evaluation", "testing", "incident", "complaint", "review"
    }

    OFF_TOPIC_KEYWORDS = {
        "marketing", "advertising", "cookie", "newsletter", "promotion", "campaign",
        "trading", "investment", "crypto", "shipping", "delivery", "warranty",
        "refund", "loyalty", "discount"
    }

    def __init__(self) -> None:
        logger.info("MockLLM initialized (test mode)")

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t for t in re.findall(r"[a-z]+", text.lower()) if len(t) >= 4}

    def _classify(self, clause: str, article: str) -> tuple[str, float, float]:
        ct = self._tokens(clause)
        at = self._tokens(article)
        if not ct or not at:
            return "NotApplicable", 0.0, 30.0

        overlap = len(ct & at) / min(len(ct), len(at))

        # Off-topic is determined by the CLAUSE only — what the policy is about,
        # not what the regulation says. A shipping clause is off-topic regardless
        # of which AI Act Article it gets compared against.
        clause_is_off_topic = (
            bool(ct & self.OFF_TOPIC_KEYWORDS)
            and not bool(ct & self.REGULATORY_KEYWORDS)
        )
        clause_is_topical = bool(ct & self.REGULATORY_KEYWORDS)

        if clause_is_off_topic:
            return "NotApplicable", 5.0, 85.0

        relevance = min(100.0, overlap * 200.0)

        if overlap >= 0.35 and clause_is_topical:
            return "Compliant", relevance, min(95.0, 60.0 + overlap * 100.0)
        if overlap >= 0.15 and clause_is_topical:
            return "Partial", relevance, min(85.0, 50.0 + overlap * 100.0)
        if clause_is_topical:
            return "Gap", max(20.0, relevance), 65.0
        return "NotApplicable", relevance, 70.0

    def _reasoning(self, clause: str, candidate: CandidateArticle, classification: str) -> str:
        if classification == "Compliant":
            return (
                f"The clause appears to satisfy {candidate.article_id} ({candidate.article_title}). "
                f"Key concepts overlap with the regulation's requirements."
            )
        if classification == "Partial":
            return (
                f"The clause touches the obligations in {candidate.article_id} but does not fully address "
                f"all requirements outlined in {candidate.article_title}."
            )
        if classification == "Gap":
            return (
                f"The clause does not adequately address the requirements of {candidate.article_id} "
                f"({candidate.article_title}). Specific elements appear to be missing."
            )
        return f"This clause is not within the scope of {candidate.article_id} ({candidate.article_title})."

    def _remediation(self, candidate: CandidateArticle, classification: str) -> str | None:
        if classification == "Partial":
            return (
                f"Strengthen the clause by adding language that explicitly references the requirements "
                f"of {candidate.article_id}, including documented evidence and ongoing review processes."
            )
        if classification == "Gap":
            return (
                f"Add a dedicated clause addressing {candidate.article_title} per {candidate.article_id}. "
                f"Include the specific evidentiary obligations and review cadence required by the article."
            )
        return None

    def map_clause(
        self,
        clause_text: str,
        heading_path: str,
        candidates: list[CandidateArticle],
    ) -> list[LLMMappingResult]:
        results = []
        for i, c in enumerate(candidates):
            classification, relevance, confidence = self._classify(clause_text, c.article_text)
            results.append(LLMMappingResult(
                candidate_index=i,
                article_id=c.article_id,
                relevance_score=relevance,
                classification=classification,
                reasoning=self._reasoning(clause_text, c, classification),
                gap_remediation=self._remediation(c, classification),
                confidence=confidence,
            ))
        return results


# ----------------- Real LLM (production) -----------------

class _GroqWithGeminiFallbackLLM:
    """Production LLM using Groq with Gemini fallback. Lazy-imported."""

    SYSTEM_PROMPT = (
        "You are a compliance mapping expert. You analyze policy clauses against regulatory "
        "Articles and produce structured JSON output. You NEVER invent Article numbers. "
        "You only cite Articles from the candidates provided. Output JSON only."
    )

    def __init__(self) -> None:
        self.settings = get_settings()
        self._groq = None
        self._gemini = None
        self._langfuse = None
        self._init_langfuse()

    def _init_langfuse(self) -> None:
        """Optional: if Langfuse keys are set, every LLM call gets traced."""
        if not (self.settings.langfuse_public_key and self.settings.langfuse_secret_key):
            return
        try:
            from langfuse import Langfuse  # type: ignore
            self._langfuse = Langfuse(
                public_key=self.settings.langfuse_public_key,
                secret_key=self.settings.langfuse_secret_key,
                host=self.settings.langfuse_host,
            )
            logger.info("Langfuse tracing enabled")
        except ImportError:
            logger.warning(
                "LANGFUSE_* keys set but `langfuse` package is not installed"
            )
        except Exception as e:  # pragma: no cover
            logger.warning(f"Langfuse init failed: {e}")

    def _ensure_groq(self):
        if self._groq is None:
            from groq import Groq  # type: ignore
            self._groq = Groq(api_key=self.settings.groq_api_key)

    def _ensure_gemini(self):
        if self._gemini is None:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=self.settings.gemini_api_key)
            self._gemini = genai.GenerativeModel(self.settings.gemini_model)

    def _build_prompt(self, clause_text: str, heading_path: str, candidates: list[CandidateArticle]) -> str:
        cand_text = "\n\n".join(
            f"{i+1}. {c.article_id} — {c.article_title}: {c.article_text[:600]}"
            for i, c in enumerate(candidates)
        )
        return (
            f"Policy Clause (from \"{heading_path}\"):\n\"{clause_text}\"\n\n"
            f"Candidate Regulation Articles:\n{cand_text}\n\n"
            'For each candidate, output an object with: candidate_index (1-based), article_id, '
            'relevance_score (0-100), classification ("Compliant"|"Partial"|"Gap"|"NotApplicable"), '
            'reasoning (1-2 sentences citing specific Article paragraphs), gap_remediation '
            '(1-2 sentences if Partial/Gap, else null), confidence (0-100).\n'
            'Output JSON with shape: {"mappings": [...]}.'
        )

    def map_clause(
        self,
        clause_text: str,
        heading_path: str,
        candidates: list[CandidateArticle],
    ) -> list[LLMMappingResult]:
        prompt = self._build_prompt(clause_text, heading_path, candidates)

        trace = None
        if self._langfuse is not None:
            try:
                trace = self._langfuse.trace(
                    name="map_clause",
                    input={"heading_path": heading_path, "clause": clause_text[:500]},
                    metadata={"candidate_count": len(candidates)},
                )
            except Exception:
                trace = None

        provider = "groq"
        try:
            self._ensure_groq()
            resp = self._groq.chat.completions.create(  # type: ignore
                model=self.settings.groq_model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"Groq failed ({e}); falling back to Gemini")
            provider = "gemini"
            try:
                self._ensure_gemini()
                resp = self._gemini.generate_content(  # type: ignore
                    self.SYSTEM_PROMPT + "\n\n" + prompt,
                    generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
                )
                content = resp.text
            except Exception as e2:
                logger.error(f"Both Groq and Gemini failed: {e2}")
                if trace is not None:
                    try:
                        trace.update(output=None, level="ERROR", status_message=str(e2))
                    except Exception:
                        pass
                return []

        if trace is not None:
            try:
                trace.generation(
                    name=provider,
                    model=self.settings.groq_model if provider == "groq" else self.settings.gemini_model,
                    output=content,
                )
            except Exception:
                pass

        try:
            data = json.loads(content or "{}")
            raw_mappings = data.get("mappings", [])
        except json.JSONDecodeError:
            logger.error("LLM returned invalid JSON")
            return []

        out = []
        for m in raw_mappings:
            try:
                out.append(LLMMappingResult(
                    candidate_index=int(m.get("candidate_index", 0)) - 1,
                    article_id=str(m.get("article_id", "")),
                    relevance_score=float(m.get("relevance_score", 0)),
                    classification=str(m.get("classification", "NotApplicable")),
                    reasoning=str(m.get("reasoning", "")),
                    gap_remediation=m.get("gap_remediation"),
                    confidence=float(m.get("confidence", 0)),
                ))
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid mapping: {e}")
                continue
        return out


_llm_singleton: LLM | None = None


def get_llm() -> LLM:
    global _llm_singleton
    if _llm_singleton is None:
        settings = get_settings()
        if settings.is_test_mode or not settings.groq_api_key:
            _llm_singleton = _MockLLM()
        else:
            _llm_singleton = _GroqWithGeminiFallbackLLM()
    return _llm_singleton
