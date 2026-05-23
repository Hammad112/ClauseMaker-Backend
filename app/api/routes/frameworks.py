"""Framework listing endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from app.core.vector_store import get_vector_store
from app.schemas import FrameworkInfo
from app.services.framework_loader import FRAMEWORKS

router = APIRouter(prefix="/api/frameworks", tags=["frameworks"])


@router.get("", response_model=list[FrameworkInfo])
async def list_frameworks():
    store = get_vector_store()
    out = []
    for spec in FRAMEWORKS.values():
        count = store.count(framework_id=spec.id)
        out.append(FrameworkInfo(
            id=spec.id,
            name=spec.name,
            description=spec.description,
            article_count=count,
            source_url=spec.source_url,
        ))
    return out
