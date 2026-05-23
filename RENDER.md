# Deploying Clausemark Backend to Render (Free Tier)

A complete walkthrough for getting the FastAPI service running on Render's
free 512 MB web service tier with real Groq LLM, real Qdrant Cloud, real
Supabase Postgres, and real Cloudflare R2.

## What this configuration gives you

| Component | Mode on free tier | Why |
| --- | --- | --- |
| LLM | **Real Groq Llama 3.3 70B** (Gemini fallback) | Hosted, no local memory cost |
| Embeddings | **Hash embedder** (in-process) | Skips PyTorch — saves ~150 MB RAM |
| Reranker | **Overlap heuristic** (in-process) | Skips bge-reranker-base — saves ~200 MB RAM |
| Vector store | **Qdrant Cloud** (free 1 GB cluster) | Persistent across restarts |
| Database | **Supabase Postgres** (free 500 MB) | Persistent across restarts |
| Object store | **Cloudflare R2** (free 10 GB, zero egress) | Persistent across restarts |
| Errors | **Sentry** (free 5k events/mo) | Optional but recommended |

The hash embedder + overlap reranker are intentional tradeoffs. The Groq LLM
does the actual classification, so retrieval quality only needs to surface the
right candidate articles — and on the 13-article EU AI Act corpus, lexical
overlap is sufficient. The end-to-end mapping quality is demo-ready.

**When to upgrade:** if your corpus grows past ~50 articles per framework,
or you need multilingual support, move to Render Starter ($7/mo, 1 GB RAM)
and flip `USE_REAL_EMBEDDER=true` + `USE_REAL_RERANKER=true`. You'll also
need to add `sentence-transformers==3.3.1` and `FlagEmbedding==1.3.4` to
`requirements-render.txt`. The vector store collection will need to be wiped
and re-ingested because the embedding dimensions/space change.

---

## 1. Provision the free external services

You can do these in any order. All require only an email signup.

### Groq (LLM)
1. <https://console.groq.com> → Sign in → API Keys → Create.
2. Copy the key (`gsk_...`) — you'll paste it into Render as `GROQ_API_KEY`.

### Google AI Studio (LLM fallback)
1. <https://aistudio.google.com> → Get API Key → Create.
2. Copy as `GEMINI_API_KEY`.

### Qdrant Cloud (vector store)
1. <https://cloud.qdrant.io> → New Cluster → Free Tier (1 GB).
2. From the cluster dashboard, copy the **HTTP URL** (e.g.
   `https://xxx.qdrant.io`) and **API key**.
3. Set `QDRANT_URL` and `QDRANT_API_KEY` in Render.

### Supabase (Postgres) — REQUIRED, not optional
> Render free tier has **no persistent disk**. If `DATABASE_URL` stays as
> SQLite, every document upload, mapping job, and result disappears the moment
> the service goes to sleep (15 minutes of idle). Demos will look broken to
> anyone who returns to the URL the next day.

1. <https://supabase.com> → New Project → free tier (500 MB, never expires).
2. Wait for the project to provision (~2 min).
3. Project Settings → Database → **Connection string** → switch the tab to
   **URI** and copy the value. Make sure to copy the *pooler* URL, not the
   direct one, so connections survive sleep.
4. Convert from `postgresql://` to `postgresql+asyncpg://` for the async
   driver. Example:
   ```
   postgresql+asyncpg://postgres.<ref>:<pw>@aws-0-eu-central-1.pooler.supabase.com:5432/postgres
   ```
5. Paste as `DATABASE_URL` in the Render dashboard.

### Cloudflare R2 (object storage)
1. Cloudflare dashboard → R2 → Create Bucket (name e.g. `clausemark-uploads`).
2. Manage R2 API Tokens → Create token (object read+write on this bucket).
3. Copy the **Access Key ID**, **Secret Access Key**, and **endpoint URL**
   shown after creation (looks like
   `https://<account>.r2.cloudflarestorage.com`).
4. Set the four `R2_*` vars in Render.

### Sentry (optional, recommended)
1. <https://sentry.io> → New Project → Python/FastAPI.
2. Copy the **DSN** → set as `SENTRY_DSN`.

---

## 2. Push to GitHub

```powershell
cd d:\ClauseMaker\ClauseMaker-Backend
git init
git add .
git commit -m "Initial Clausemark backend + frontend"
# Create a repo on github.com, then:
git remote add origin https://github.com/<you>/clausemark.git
git branch -M main
git push -u origin main
```

The backend lives at `clausemark-backend/`. The `render.yaml` is configured
with `rootDir: clausemark-backend` so Render builds only that subtree.

---

## 3. Create the Render service

1. <https://dashboard.render.com> → New → Blueprint.
2. Connect your GitHub account, pick the `clausemark` repo.
3. Render reads `clausemark-backend/render.yaml` automatically. Confirm:
   - Plan: **Free**
   - Build command: `pip install --upgrade pip && pip install -r requirements-render.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Health check path: `/health`
4. In the Environment section, paste in the secrets you collected above:
   `GROQ_API_KEY`, `GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`,
   `DATABASE_URL`, `R2_ACCESS_KEY`, `R2_SECRET_KEY`, `R2_BUCKET`,
   `R2_ENDPOINT`, `SENTRY_DSN` (optional), and `CORS_ORIGINS` (set this to
   your deployed frontend origin, comma-separated).
5. Click **Apply / Create Web Service**. First build takes ~4 minutes.

When the build is green, Render assigns a URL like
`https://clausemark-backend.onrender.com`.

---

## 4. Smoke test

```powershell
curl https://clausemark-backend.onrender.com/health
# → {"status":"ok"}

curl https://clausemark-backend.onrender.com/api/frameworks
# → [{"id":"eu_ai_act","name":"EU AI Act","article_count":13, ...}, ...]
```

Watch the build logs — you should see:

```
Starting Clausemark backend in PRODUCTION mode
Sentry initialized            ← if you set SENTRY_DSN
Qdrant: connecting to https://...qdrant.io
R2ObjectStore bucket=clausemark-uploads
Loaded 13 articles for framework eu_ai_act
Framework articles loaded: {'eu_ai_act': 13, 'gdpr': 2}
```

If you see warnings like `DATABASE_URL is SQLite — data will be LOST on
restart`, your secrets aren't all set yet.

---

## 5. Wire the frontend

In your Vercel (or Render static) project for `clausemark-frontend`, set:

```
NEXT_PUBLIC_API_URL=https://clausemark-backend.onrender.com
```

Redeploy the frontend. The upload + results flow now hits your live backend.

Also add the frontend's origin to the backend's `CORS_ORIGINS` env var (e.g.
`https://clausemark.vercel.app`) and redeploy the backend, or browsers will
block requests.

---

## 6. Keep it warm

Render free web services sleep after 15 minutes of inactivity (~30–60 s cold
start). For demos:

- <https://uptimerobot.com> → free 50 monitors → HTTP check on
  `https://clausemark-backend.onrender.com/health` every 5 minutes.
- Or <https://cron-job.org> → unlimited free crons → same URL every 10 min.

---

## Limitations of the free tier configuration

- **Retrieval quality is heuristic.** With more than ~30 articles per framework
  the hash embedder starts to mis-rank candidates. Upgrade path: enable BGE.
- **No OCR.** Image-only scanned PDFs are rejected with a clear message
  pointing the user to paste text. Tesseract integration is out of scope for
  the MVP per the project spec.
- **No Langfuse wrap.** The schema and env vars are in place but LLM calls are
  not currently traced through Langfuse. Easy follow-up.
- **No Resend email gate.** `EmailGateRequest` schema exists but no route.
  Add when you start the outbound demo cadence.
- **No EUR-Lex auto-ingestion.** The curated 13-article corpus ships in code
  (`app/services/framework_loader.py`). Full HTML scraper + quarterly
  cron-job.org refresh is documented in the spec but not yet built.

---

## Troubleshooting

**Build fails on `pip install`**
Check the Render build log. The Python version is pinned to 3.11.9 via the
`PYTHON_VERSION` env var. If a dep can't resolve, pin a stricter version in
`requirements-render.txt`.

**Service boots but `/api/frameworks` is empty**
The startup hook calls `ensure_frameworks_loaded()`. If Qdrant is unreachable
the call fails. Verify `QDRANT_URL` (must start with `https://` and end with
the cluster suffix — not the dashboard URL).

**`502 Bad Gateway` on first request after sleep**
Cold start in progress. Wait 30–60 s and retry, or attach an uptime pinger
(step 6).

**OOM / SIGKILL in logs**
You probably enabled `USE_REAL_EMBEDDER=true` on the free tier. Set it back
to `false`, or upgrade to Render Starter.

**CORS errors in browser console**
`CORS_ORIGINS` is empty or doesn't include the frontend origin. It's a
comma-separated list — e.g. `https://app.example.com,http://localhost:3000`.

**"Index required but not found for framework_id"**
Qdrant Cloud requires a payload index for filtered count/search. The backend
creates it automatically on startup (`_ensure_collection`). If you see this,
your code is older than the fix in `app/core/vector_store.py`. Redeploy.

**"Not existing vector name error" on upsert**
Your Qdrant collection was pre-created (via the Qdrant Cloud UI) with a
schema that doesn't match — usually named vectors with a larger dimension.
The backend now detects this and recreates the collection automatically on
boot. Watch the logs for the line:
> `Qdrant collection 'X' has an incompatible vector schema (...). Recreating
> with unnamed 384-d cosine vectors.`
