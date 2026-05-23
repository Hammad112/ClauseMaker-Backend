"""PDF report export."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.db import Clause, MappingJob, Document, get_session_maker
from app.services.confidence_scorer import primary_status_for_clause
from app.services.framework_loader import FRAMEWORKS
from app.services.report_generator import generate_audit_pdf

router = APIRouter(prefix="/api/reports", tags=["reports"])


async def _session() -> AsyncSession:
    sm = get_session_maker()
    async with sm() as s:
        yield s


@router.post("/{job_id}/export")
async def export_audit_pdf(
    job_id: str,
    company_name: str = "Acme Corporation",
    session: AsyncSession = Depends(_session),
):
    job = await session.get(MappingJob, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "done":
        raise HTTPException(409, "Job not complete")

    document = await session.get(Document, job.document_id)

    stmt = (
        select(Clause)
        .where(Clause.job_id == job_id)
        .options(selectinload(Clause.mappings))
        .order_by(Clause.position)
    )
    result = await session.execute(stmt)
    clauses_db = result.scalars().all()

    framework_name = FRAMEWORKS[job.framework_id].name if job.framework_id in FRAMEWORKS else job.framework_id

    clauses_data = []
    for c in clauses_db:
        primary, conf = primary_status_for_clause(
            [(m.classification, m.confidence) for m in c.mappings]
        )
        clauses_data.append({
            "position": c.position,
            "heading_path": c.heading_path,
            "text": c.text,
            "primary_status": primary,
            "primary_confidence": conf,
            "mappings": [
                {
                    "article_id": m.article_id,
                    "article_title": m.article_title,
                    "article_text": m.article_text,
                    "classification": m.classification,
                    "confidence": m.confidence,
                    "reasoning": m.reasoning,
                    "gap_remediation": m.gap_remediation,
                    "source_url": m.source_url,
                }
                for m in c.mappings
            ],
        })

    job_data = {
        "id": job.id,
        "total_clauses": job.total_clauses,
        "compliant_count": job.compliant_count,
        "partial_count": job.partial_count,
        "gap_count": job.gap_count,
        "not_applicable_count": job.not_applicable_count,
        "average_confidence": job.average_confidence,
    }

    pdf_bytes = generate_audit_pdf(
        job_data=job_data,
        clauses_data=clauses_data,
        framework_name=framework_name,
        company_name=company_name,
    )

    filename = f"clausemark_{framework_name.replace(' ', '_')}_{job_id[:8]}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
