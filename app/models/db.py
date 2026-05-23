"""SQLAlchemy ORM models."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import String, Integer, Float, Text, ForeignKey, DateTime, JSON
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.core.config import get_settings


def _uuid() -> str:
    return str(uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    content_type: Mapped[str] = mapped_column(String, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_key: Mapped[str] = mapped_column(String, nullable=False)
    uploader_ip: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    mapping_jobs: Mapped[list[MappingJob]] = relationship(back_populates="document")


class MappingJob(Base):
    __tablename__ = "mapping_jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), nullable=False)
    framework_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="queued")  # queued|parsing|extracting|mapping|scoring|done|failed
    stage_message: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    total_clauses: Mapped[int] = mapped_column(Integer, default=0)
    completed_clauses: Mapped[int] = mapped_column(Integer, default=0)
    compliant_count: Mapped[int] = mapped_column(Integer, default=0)
    partial_count: Mapped[int] = mapped_column(Integer, default=0)
    gap_count: Mapped[int] = mapped_column(Integer, default=0)
    not_applicable_count: Mapped[int] = mapped_column(Integer, default=0)
    average_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    document: Mapped[Document] = relationship(back_populates="mapping_jobs")
    clauses: Mapped[list[Clause]] = relationship(back_populates="job", cascade="all, delete-orphan")


class Clause(Base):
    __tablename__ = "clauses"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("mapping_jobs.id"), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    heading_path: Mapped[str] = mapped_column(String, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, default=0)

    job: Mapped[MappingJob] = relationship(back_populates="clauses")
    mappings: Mapped[list[ClauseMapping]] = relationship(back_populates="clause", cascade="all, delete-orphan")


class ClauseMapping(Base):
    __tablename__ = "clause_mappings"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    clause_id: Mapped[str] = mapped_column(ForeignKey("clauses.id"), nullable=False)
    article_id: Mapped[str] = mapped_column(String, nullable=False)
    article_title: Mapped[str] = mapped_column(String, nullable=False)
    article_text: Mapped[str] = mapped_column(Text, nullable=False)
    framework_id: Mapped[str] = mapped_column(String, nullable=False)
    classification: Mapped[str] = mapped_column(String, nullable=False)  # Compliant|Partial|Gap|NotApplicable
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    retrieval_score: Mapped[float] = mapped_column(Float, default=0.0)
    reranker_score: Mapped[float] = mapped_column(Float, default=0.0)
    llm_self_score: Mapped[float] = mapped_column(Float, default=0.0)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    gap_remediation: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_url: Mapped[str | None] = mapped_column(String, nullable=True)

    clause: Mapped[Clause] = relationship(back_populates="mappings")


# ------- Session helpers -------

_engine = None
_session_maker: async_sessionmaker[AsyncSession] | None = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        url = settings.database_url
        connect_args: dict = {}
        # Supabase transaction-mode pooler (port 6543) doesn't support prepared
        # statements. asyncpg caches them by default, so the second query fails
        # with "prepared statement '__asyncpg_stmt_X__' does not exist". Disable
        # the cache for any pooler URL.
        if "+asyncpg" in url and ("pooler.supabase" in url or ":6543" in url):
            connect_args["statement_cache_size"] = 0
            connect_args["prepared_statement_cache_size"] = 0
        _engine = create_async_engine(url, future=True, connect_args=connect_args)
    return _engine


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    global _session_maker
    if _session_maker is None:
        _session_maker = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_maker


async def init_db() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def reset_db() -> None:
    """Drop and recreate all tables — useful for tests."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
