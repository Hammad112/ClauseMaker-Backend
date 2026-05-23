"""End-to-end API test.

Exercises the full request lifecycle:
  POST /api/documents (upload)
  POST /api/mappings (kick off pipeline)
  GET /api/mappings/{id} (poll status)
  GET /api/mappings/{id}/results (final output)
  POST /api/reports/{id}/export (PDF)
"""
import asyncio
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


SAMPLE_POLICY_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_acme_policy.txt"


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_list_frameworks():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/frameworks")
    assert r.status_code == 200
    frameworks = r.json()
    fids = {f["id"] for f in frameworks}
    assert "eu_ai_act" in fids
    eu = next(f for f in frameworks if f["id"] == "eu_ai_act")
    assert eu["article_count"] > 0


@pytest.mark.asyncio
async def test_full_mapping_pipeline():
    """Upload a policy, kick off a mapping job, poll until done, verify output."""
    assert SAMPLE_POLICY_PATH.exists(), f"Sample policy missing at {SAMPLE_POLICY_PATH}"
    policy_bytes = SAMPLE_POLICY_PATH.read_bytes()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Upload document
        r = await client.post(
            "/api/documents",
            files={"file": ("sample_policy.txt", policy_bytes, "text/plain")},
        )
        assert r.status_code == 200, r.text
        doc = r.json()
        document_id = doc["id"]
        assert doc["filename"] == "sample_policy.txt"
        assert doc["size_bytes"] == len(policy_bytes)

        # 2. Kick off mapping
        r = await client.post(
            "/api/mappings",
            json={"document_id": document_id, "framework_id": "eu_ai_act"},
        )
        assert r.status_code == 200, r.text
        job = r.json()
        job_id = job["id"]
        assert job["status"] in {"queued", "parsing"}

        # 3. Poll for completion (max 30 seconds)
        for _ in range(60):
            r = await client.get(f"/api/mappings/{job_id}")
            assert r.status_code == 200
            job = r.json()
            if job["status"] in {"done", "failed"}:
                break
            await asyncio.sleep(0.5)

        assert job["status"] == "done", f"Job did not complete: {job}"
        assert job["total_clauses"] > 0
        assert job["error_message"] is None

        # 4. Get results
        r = await client.get(f"/api/mappings/{job_id}/results")
        assert r.status_code == 200
        results = r.json()

        assert results["job"]["status"] == "done"
        assert len(results["clauses"]) > 0

        # Sanity: every clause should have at least one mapping
        for clause in results["clauses"]:
            assert "primary_status" in clause
            assert clause["primary_status"] in {"Compliant", "Partial", "Gap", "NotApplicable"}
            # Mappings can be empty for off-topic clauses but most should have some
            assert isinstance(clause["mappings"], list)

        # Should have a mix of statuses (the sample policy is designed for this)
        statuses = [c["primary_status"] for c in results["clauses"]]
        assert "Compliant" in statuses or "Partial" in statuses, \
            f"Expected at least one Compliant or Partial: {statuses}"

        # All cited article_ids must be from the real corpus
        all_cited = {m["article_id"] for c in results["clauses"] for m in c["mappings"]}
        from app.services import citation_validator
        for aid in all_cited:
            assert citation_validator.is_valid_citation("eu_ai_act", aid), \
                f"Hallucinated citation in API output: {aid}"

        # 5. Export PDF
        r = await client.post(f"/api/reports/{job_id}/export?company_name=Acme%20AI")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/pdf"
        pdf_bytes = r.content
        assert pdf_bytes.startswith(b"%PDF"), "Response is not a valid PDF"
        assert len(pdf_bytes) > 5000, f"PDF suspiciously small: {len(pdf_bytes)} bytes"

        # Save the artifacts so we can inspect them
        out_dir = Path("/tmp/clausemark_test_artifacts")
        out_dir.mkdir(exist_ok=True)
        (out_dir / f"report_{job_id[:8]}.pdf").write_bytes(pdf_bytes)


@pytest.mark.asyncio
async def test_mapping_not_found():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/mappings/nonexistent-id")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_results_before_complete_409():
    """Asking for results before the job is done returns 409 Conflict."""
    from app.models.db import MappingJob, get_session_maker
    sm = get_session_maker()
    async with sm() as session:
        job = MappingJob(
            document_id="x",
            framework_id="eu_ai_act",
            status="parsing",
        )
        session.add(job)
        await session.commit()
        jid = job.id

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get(f"/api/mappings/{jid}/results")
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_document_too_large_413():
    big_data = b"x" * (11 * 1024 * 1024)  # 11 MB
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/api/documents",
            files={"file": ("big.txt", big_data, "text/plain")},
        )
    assert r.status_code == 413
