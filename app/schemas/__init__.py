"""Pydantic v2 schemas for API I/O."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Classification = Literal["Compliant", "Partial", "Gap", "NotApplicable"]
JobStatus = Literal["queued", "parsing", "extracting", "mapping", "scoring", "done", "failed"]


class DocumentOut(BaseModel):
    id: str
    filename: str
    content_type: str
    size_bytes: int
    created_at: datetime


class FrameworkInfo(BaseModel):
    id: str
    name: str
    description: str
    article_count: int
    source_url: str


class MappingRequest(BaseModel):
    document_id: str
    framework_id: str = Field(default="eu_ai_act")


class MappingJobOut(BaseModel):
    id: str
    document_id: str
    framework_id: str
    status: JobStatus
    stage_message: str | None = None
    total_clauses: int
    completed_clauses: int
    compliant_count: int
    partial_count: int
    gap_count: int
    not_applicable_count: int
    average_confidence: float
    created_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None


class ClauseMappingOut(BaseModel):
    article_id: str
    article_title: str
    article_text: str
    framework_id: str
    classification: Classification
    confidence: float
    retrieval_score: float
    reranker_score: float
    llm_self_score: float
    reasoning: str | None
    gap_remediation: str | None
    source_url: str | None


class ClauseOut(BaseModel):
    id: str
    position: int
    heading_path: str
    text: str
    primary_status: Classification
    primary_confidence: float
    mappings: list[ClauseMappingOut]


class MappingResultsOut(BaseModel):
    job: MappingJobOut
    clauses: list[ClauseOut]


class EmailGateRequest(BaseModel):
    email: str
    job_id: str
