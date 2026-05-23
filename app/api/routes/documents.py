"""Document upload and management endpoints."""
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.object_store import get_object_store, storage_key
from app.models.db import Document, get_session_maker
from app.schemas import DocumentOut

router = APIRouter(prefix="/api/documents", tags=["documents"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
}


async def _session() -> AsyncSession:
    sm = get_session_maker()
    async with sm() as s:
        yield s


@router.post("", response_model=DocumentOut)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(_session),
):
    # Read file with size guard
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES} bytes)")
    if not data:
        raise HTTPException(400, "Empty file")

    content_type = file.content_type or "application/octet-stream"

    # Persist to object store
    obj_store = get_object_store()
    key = storage_key(file.filename or "uploaded")
    obj_store.put(key, data, content_type)

    client_ip = request.client.host if request.client else None
    doc = Document(
        filename=file.filename or "uploaded",
        content_type=content_type,
        size_bytes=len(data),
        storage_key=key,
        uploader_ip=client_ip,
    )
    session.add(doc)
    await session.commit()
    await session.refresh(doc)

    return DocumentOut(
        id=doc.id,
        filename=doc.filename,
        content_type=doc.content_type,
        size_bytes=doc.size_bytes,
        created_at=doc.created_at,
    )


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(document_id: str, session: AsyncSession = Depends(_session)):
    doc = await session.get(Document, document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return DocumentOut(
        id=doc.id,
        filename=doc.filename,
        content_type=doc.content_type,
        size_bytes=doc.size_bytes,
        created_at=doc.created_at,
    )
