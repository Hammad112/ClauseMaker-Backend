"""Mapping job endpoints — the heart of the API."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.object_store import get_object_store
from app.models.db import Clause, ClauseMapping, Document, MappingJob, get_session_maker
from app.schemas import (
    ClauseMappingOut, ClauseOut, MappingJobOut, MappingRequest, MappingResultsOut,
)
from app.services import clause_mapper
from app.services.confidence_scorer import primary_status_for_clause
from app.services.document_parser import (
    ScannedPDFError,
    extract_clauses,
    parse_document,
)

router = APIRouter(prefix="/api/mappings", tags=["mappings"])


async def _session() -> AsyncSession:
    sm = get_session_maker()
    async with sm() as s:
        yield s


def _to_job_schema(job: MappingJob) -> MappingJobOut:
    return MappingJobOut(
        id=job.id,
        document_id=job.document_id,
        framework_id=job.framework_id,
        status=job.status,  # type: ignore
        stage_message=job.stage_message,
        total_clauses=job.total_clauses,
        completed_clauses=job.completed_clauses,
        compliant_count=job.compliant_count,
        partial_count=job.partial_count,
        gap_count=job.gap_count,
        not_applicable_count=job.not_applicable_count,
        average_confidence=job.average_confidence,
        created_at=job.created_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
    )


@router.post("", response_model=MappingJobOut)
async def create_mapping(
    req: MappingRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(_session),
):
    doc = await session.get(Document, req.document_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    job = MappingJob(
        document_id=req.document_id,
        framework_id=req.framework_id,
        status="queued",
        stage_message="Queued",
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    background_tasks.add_task(_run_mapping_pipeline, job.id)

    return _to_job_schema(job)


@router.get("/{job_id}", response_model=MappingJobOut)
async def get_mapping_status(job_id: str, session: AsyncSession = Depends(_session)):
    job = await session.get(MappingJob, job_id)
    if not job:
        raise HTTPException(404, "Mapping job not found")
    return _to_job_schema(job)


@router.get("/{job_id}/results", response_model=MappingResultsOut)
async def get_mapping_results(job_id: str, session: AsyncSession = Depends(_session)):
    job = await session.get(MappingJob, job_id)
    if not job:
        raise HTTPException(404, "Mapping job not found")
    if job.status != "done":
        raise HTTPException(409, f"Job not complete (status={job.status})")

    stmt = (
        select(Clause)
        .where(Clause.job_id == job_id)
        .options(selectinload(Clause.mappings))
        .order_by(Clause.position)
    )
    result = await session.execute(stmt)
    clauses_db = result.scalars().all()

    clause_outs = []
    for c in clauses_db:
        mappings_out = [ClauseMappingOut(
            article_id=m.article_id,
            article_title=m.article_title,
            article_text=m.article_text,
            framework_id=m.framework_id,
            classification=m.classification,  # type: ignore
            confidence=m.confidence,
            retrieval_score=m.retrieval_score,
            reranker_score=m.reranker_score,
            llm_self_score=m.llm_self_score,
            reasoning=m.reasoning,
            gap_remediation=m.gap_remediation,
            source_url=m.source_url,
        ) for m in c.mappings]

        # Determine primary status for this clause
        primary, conf = primary_status_for_clause(
            [(m.classification, m.confidence) for m in c.mappings]
        )

        clause_outs.append(ClauseOut(
            id=c.id,
            position=c.position,
            heading_path=c.heading_path,
            text=c.text,
            primary_status=primary,  # type: ignore
            primary_confidence=conf,
            mappings=mappings_out,
        ))

    return MappingResultsOut(job=_to_job_schema(job), clauses=clause_outs)


# ----------------- Background pipeline -----------------

async def _run_mapping_pipeline(job_id: str) -> None:
    """The async background task that runs the full mapping pipeline."""
    sm = get_session_maker()
    async with sm() as session:
        job = await session.get(MappingJob, job_id)
        if not job:
            logger.error(f"Job {job_id} disappeared")
            return

        try:
            # Stage 1: parse
            job.status = "parsing"
            job.stage_message = "Parsing document"
            await session.commit()

            doc = await session.get(Document, job.document_id)
            if not doc:
                raise RuntimeError("Document not found")

            obj_store = get_object_store()
            raw_bytes = obj_store.get(doc.storage_key)
            try:
                raw_text = parse_document(doc.filename, raw_bytes, doc.content_type)
            except ScannedPDFError as e:
                job.status = "failed"
                job.error_message = str(e)
                job.completed_at = datetime.now(timezone.utc)
                await session.commit()
                return

            # Stage 2: extract clauses
            job.status = "extracting"
            job.stage_message = "Extracting clauses"
            await session.commit()

            parsed_clauses = extract_clauses(raw_text)
            job.total_clauses = len(parsed_clauses)
            await session.commit()

            if not parsed_clauses:
                job.status = "failed"
                job.error_message = (
                    "No clauses could be extracted from this document. If it's a "
                    "scanned/image-only PDF, OCR is not supported in this version — "
                    "please paste the text into a .txt file or upload a text-based "
                    "PDF or .docx instead."
                )
                job.completed_at = datetime.now(timezone.utc)
                await session.commit()
                return

            # Persist clause rows
            clause_rows: list[Clause] = []
            for pc in parsed_clauses:
                row = Clause(
                    job_id=job.id,
                    position=pc.position,
                    heading_path=pc.heading_path,
                    text=pc.text,
                    char_count=pc.char_count,
                )
                session.add(row)
                clause_rows.append(row)
            await session.commit()
            for row in clause_rows:
                await session.refresh(row)

            # Stage 3 & 4: map (retrieve + rerank + LLM + validate + score)
            job.status = "mapping"
            job.stage_message = "Mapping clauses to articles"
            await session.commit()

            clauses_for_mapper = [(pc.heading_path, pc.text) for pc in parsed_clauses]
            mapping_results = clause_mapper.map_clauses(
                clauses=clauses_for_mapper,
                framework_id=job.framework_id,
            )

            # Stage 5: persist + summarize
            job.status = "scoring"
            job.stage_message = "Computing confidence and summary"
            await session.commit()

            compliant = partial = gap = na = 0
            conf_sum = 0.0
            primary_count = 0

            for clause_row, results in zip(clause_rows, mapping_results):
                if not results:
                    na += 1
                    continue
                for r in results:
                    session.add(ClauseMapping(
                        clause_id=clause_row.id,
                        article_id=r.article_id,
                        article_title=r.article_title,
                        article_text=r.article_text,
                        framework_id=r.framework_id,
                        classification=r.classification,
                        confidence=r.confidence,
                        retrieval_score=r.retrieval_score,
                        reranker_score=r.reranker_score,
                        llm_self_score=r.llm_self_score,
                        reasoning=r.reasoning,
                        gap_remediation=r.gap_remediation,
                        source_url=r.source_url,
                    ))
                primary, conf = primary_status_for_clause(
                    [(r.classification, r.confidence) for r in results]
                )
                if primary == "Compliant":
                    compliant += 1
                elif primary == "Partial":
                    partial += 1
                elif primary == "Gap":
                    gap += 1
                else:
                    na += 1
                conf_sum += conf
                primary_count += 1
                job.completed_clauses = primary_count

            job.compliant_count = compliant
            job.partial_count = partial
            job.gap_count = gap
            job.not_applicable_count = na
            job.average_confidence = round(conf_sum / primary_count, 1) if primary_count else 0.0
            job.status = "done"
            job.stage_message = "Complete"
            job.completed_at = datetime.now(timezone.utc)
            await session.commit()

            logger.info(
                f"Job {job_id} complete: total={job.total_clauses} "
                f"compliant={compliant} partial={partial} gap={gap} na={na} "
                f"avg_conf={job.average_confidence}"
            )
        except Exception as e:  # pragma: no cover (defensive catch-all)
            logger.exception(f"Mapping pipeline failed for job {job_id}")
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc)
            await session.commit()
