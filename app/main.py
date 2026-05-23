"""Clausemark backend — FastAPI app entry point."""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import documents, frameworks, health, mappings, reports
from app.core.config import get_settings
from app.models.db import init_db
from app.services.framework_loader import ensure_frameworks_loaded


def _init_sentry(dsn: str) -> None:
    """Initialize Sentry if a DSN is configured. No-op otherwise."""
    if not dsn:
        return
    try:
        import sentry_sdk  # type: ignore
        from sentry_sdk.integrations.fastapi import FastApiIntegration  # type: ignore

        sentry_sdk.init(
            dsn=dsn,
            integrations=[FastApiIntegration()],
            traces_sample_rate=0.1,
            send_default_pii=False,
        )
        logger.info("Sentry initialized")
    except ImportError:
        logger.warning("SENTRY_DSN set but sentry-sdk is not installed")
    except Exception as e:  # pragma: no cover (defensive)
        logger.warning(f"Sentry init failed: {e}")


def _warn_ephemeral_storage(settings) -> None:
    """Render free-tier has no persistent disk — warn loudly if local fallbacks
    are in play in production mode so we don't lose data silently on restart."""
    if settings.is_test_mode:
        return
    if settings.database_url.startswith("sqlite"):
        logger.warning(
            "DATABASE_URL is SQLite — data will be LOST on restart. "
            "Set DATABASE_URL to a Supabase Postgres URL for persistence."
        )
    if not settings.qdrant_url:
        logger.warning(
            "QDRANT_URL is empty — vector store will run in-memory and reset "
            "on every restart. Set QDRANT_URL to a Qdrant Cloud cluster for persistence."
        )
    if not settings.r2_access_key:
        logger.warning(
            "R2 credentials missing — uploads go to local ./uploads/ which is "
            "ephemeral on Render free tier. Configure Cloudflare R2 for persistence."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.info(f"Starting Clausemark backend in {settings.app_mode.upper()} mode")

    _init_sentry(settings.sentry_dsn)
    _warn_ephemeral_storage(settings)

    # Initialize DB tables
    await init_db()

    # Load framework articles into vector store
    counts = ensure_frameworks_loaded()
    logger.info(f"Framework articles loaded: {counts}")

    yield

    logger.info("Clausemark backend shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Clausemark API",
        version="1.0.0",
        description="AI Governance & Compliance Mapping",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list or ["*"],
        allow_origin_regex=settings.cors_origin_regex or None,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(frameworks.router)
    app.include_router(mappings.router)
    app.include_router(reports.router)

    @app.get("/")
    async def root():
        return {
            "service": "clausemark-backend",
            "version": "1.0.0",
            "mode": settings.app_mode,
            "docs": "/docs",
        }

    return app


app = create_app()
