# Clausemark Backend

AI governance and compliance mapping platform. Takes a policy document (PDF/DOCX/text) and maps every clause to specific Articles of regulatory frameworks (EU AI Act, GDPR, ...). Every mapping is cited, confidence-scored, and an audit-ready PDF can be exported.

## What this repo contains

Production-grade FastAPI backend with:

- **Document parsing** — PDF (pypdf), DOCX (python-docx), text; clause boundary detection by numbered/lettered/bullet markers with heading breadcrumbs
- **Hybrid retrieval pipeline** — embed → Qdrant vector search → cross-encoder rerank → LLM mapping
- **Anti-hallucination citation validator** — every Article ID cited by the LLM is checked against the indexed corpus before being returned to the client
- **Confidence fusion** — 0.3 retrieval similarity + 0.3 reranker score + 0.4 LLM self-assessment
- **Audit PDF export** — reportlab cover page, executive summary, per-clause breakdown, citation appendix
- **Test mode** — entire pipeline runs locally with in-memory mocks for LLM/embedder/vector store

## Quick start (local, test mode)

```bash
pip install -r requirements.txt
cp .env.example .env   # APP_MODE=test by default
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the interactive API.

### Run tests

```bash
pytest -v
```

33 tests cover the document parser, citation validator, confidence scorer, clause mapper, and full HTTP API end-to-end via `httpx.AsyncClient`.

### Run the demo pipeline

```bash
python scripts/demo_pipeline.py
```

Runs the full pipeline on `data/sample_acme_policy.txt` and prints color-coded mappings.

## Switching to production

In `.env` set:

```bash
APP_MODE=production
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIza...           # fallback for Groq 429s
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=...
DATABASE_URL=postgresql+asyncpg://user:pass@host/db   # Supabase
R2_ACCESS_KEY=...
R2_SECRET_KEY=...
R2_BUCKET=clausemark-uploads
R2_ENDPOINT=https://....r2.cloudflarestorage.com
CORS_ORIGINS=https://clausemark.com
```

Then uncomment the optional production deps in `requirements.txt` (groq, google-generativeai, sentence-transformers, FlagEmbedding, boto3, resend, langfuse) and `pip install -r requirements.txt` again.

## Deploy to Render

```bash
# Push to GitHub, then in Render:
# New → Web Service → Connect repo → render.yaml is auto-detected
# Add the secret env vars in the dashboard (they're marked sync:false)
```

The included `render.yaml` configures the free 512 MB tier. The backend's runtime memory budget is ~350 MB with the real BGE embedder + reranker loaded, leaving headroom for request processing.

To prevent cold starts during sales calls, configure UptimeRobot to hit `/health` every 5–10 minutes.

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness probe for UptimeRobot |
| GET | `/api/frameworks` | List indexed frameworks with article counts |
| POST | `/api/documents` | Upload policy (multipart, max 10 MB) |
| GET | `/api/documents/{id}` | Document metadata |
| POST | `/api/mappings` | Kick off mapping job (returns job_id) |
| GET | `/api/mappings/{job_id}` | Job status + progress |
| GET | `/api/mappings/{job_id}/results` | Full results (404 until status=done) |
| POST | `/api/reports/{job_id}/export` | Generate and download audit PDF |

Full OpenAPI spec at `/docs` when the server is running.

## Architecture

```
Upload → Parse (pypdf/python-docx) → Chunk (clause boundaries with heading breadcrumbs)
       → Embed (BGE-small) → Qdrant retrieve top-15
       → Reranker top-5 → LLM map (Groq primary, Gemini fallback)
       → Citation validate (drop hallucinations) → Fuse confidence
       → Persist (Postgres) → Return / Export PDF
```

Every external dependency is a swappable provider:

| Module | Test mode | Production |
|--------|-----------|------------|
| `app/core/embeddings.py` | HashEmbedder | BAAI/bge-small-en-v1.5 |
| `app/core/reranker.py` | OverlapReranker | BAAI/bge-reranker-base |
| `app/core/vector_store.py` | QdrantClient(":memory:") | Qdrant Cloud |
| `app/core/llm.py` | MockLLM (rule-based) | Groq Llama 3.3 70B + Gemini fallback |
| `app/core/object_store.py` | Local filesystem | Cloudflare R2 |
| `DATABASE_URL` | SQLite in-file | Supabase Postgres |

Toggle is just the `APP_MODE` env var. Same code path either way.

## Project layout

```
clausemark-backend/
├── app/
│   ├── api/routes/        # documents, frameworks, mappings, reports, health
│   ├── core/              # config, llm, embeddings, reranker, vector_store, object_store
│   ├── services/          # document_parser, clause_mapper, citation_validator,
│   │                       # confidence_scorer, framework_loader, report_generator
│   ├── models/db.py       # SQLAlchemy ORM
│   ├── schemas/           # Pydantic v2 I/O schemas
│   └── main.py            # FastAPI app entry point
├── scripts/
│   └── demo_pipeline.py   # Walks the full pipeline + prints color-coded output
├── tests/                 # 33 tests; pytest --tb=short
├── data/
│   └── sample_acme_policy.txt   # Demo policy (14+ clauses, designed mix)
├── Dockerfile
├── render.yaml
├── requirements.txt
└── .env.example
```

## License

Proprietary. © Hammad Nasir / hammadnasir797@gmail.com

