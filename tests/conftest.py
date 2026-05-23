"""Test configuration. Forces test mode and clean DB for every test session."""
import os
import sys
from pathlib import Path

# Ensure tests always run in test mode regardless of .env
os.environ["APP_MODE"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_clausemark.db"

# Clean previous test DB
db_file = Path("./test_clausemark.db")
if db_file.exists():
    db_file.unlink()

# Make app importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest_asyncio  # noqa: E402

from app.core.embeddings import get_embedder  # noqa: E402
from app.core.vector_store import get_vector_store, reset_vector_store_singleton  # noqa: E402
from app.models.db import reset_db  # noqa: E402
from app.services import citation_validator  # noqa: E402
from app.services.framework_loader import ensure_frameworks_loaded  # noqa: E402


@pytest_asyncio.fixture(scope="session", autouse=True)
async def _init_test_environment():
    """Initialize DB and load frameworks once per test session."""
    await reset_db()
    # Ensure vector store starts fresh
    reset_vector_store_singleton()
    # Warm up embedder so first test isn't slow
    get_embedder().encode(["warmup"])
    # Load frameworks
    counts = ensure_frameworks_loaded()
    citation_validator.refresh_cache()
    assert sum(counts.values()) > 0, "No framework articles loaded"
    yield
