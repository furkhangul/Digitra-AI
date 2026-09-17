from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Digitra API configuration.

    Values are read from environment variables / .env. No secret ever has a
    hardcoded default here — see .env.example for the expected keys.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Digitra API"
    environment: str = "development"

    cors_allow_origins: list[str] = ["http://localhost:3000"]

    rate_limit_per_minute: int = 60

    active_model_version: str = "digitra-tid-robust-v5.0.0"
    models_dir: str = "../../models"

    stt_provider: str = "browser"
    tts_provider: str = "browser"
    cloud_stt_api_key: str | None = None
    cloud_tts_api_key: str | None = None
    llm_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
