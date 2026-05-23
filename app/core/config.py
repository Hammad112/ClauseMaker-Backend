"""Application configuration loaded from environment variables."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Mode
    app_mode: str = "test"  # "test" or "production"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:3000"
    # Optional regex (Python re) that, if set, allows any origin matching it
    # in addition to the explicit list in cors_origins. Use this on Render so
    # every Vercel preview-deploy URL passes CORS without updating env vars
    # on every push. Example: https://.*\.vercel\.app
    cors_origin_regex: str = ""

    # Database
    database_url: str = "sqlite+aiosqlite:///./clausemark.db"

    # LLM
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # Vector store
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "clausemark_frameworks"

    # Object store
    r2_access_key: str = ""
    r2_secret_key: str = ""
    r2_bucket: str = "clausemark-uploads"
    r2_endpoint: str = ""
    r2_region: str = "auto"

    # Observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    sentry_dsn: str = ""

    # Email
    resend_api_key: str = ""
    resend_from: str = "noreply@clausemark.com"

    # Rate limiting
    demo_mappings_per_ip_per_day: int = 1

    # Granular component switches.
    # In test mode everything is mocked. In production, these gate whether the
    # heavy ML models are loaded. Default OFF so the service fits comfortably in
    # Render's 512 MB free tier. Set USE_REAL_EMBEDDER=true / USE_REAL_RERANKER=true
    # on a host with >= 1 GB RAM (e.g. Render Starter).
    use_real_embedder: bool = False
    use_real_reranker: bool = False

    @property
    def is_test_mode(self) -> bool:
        return self.app_mode.lower() == "test"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
