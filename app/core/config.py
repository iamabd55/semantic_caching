from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "Semantic Caching Gateway"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # LLM API Keys (loaded from .env)
    OPENROUTER_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None

    # Infrastructure URLs
    QDRANT_URL: str = "http://localhost:6333"
    REDIS_URL: str = "redis://localhost:6379"

    # Semantic Cache Defaults
    SIMILARITY_THRESHOLD: float = 0.85

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
