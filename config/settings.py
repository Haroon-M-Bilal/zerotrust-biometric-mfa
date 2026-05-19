"""
Centralized configuration using Pydantic Settings.
All environment-dependent values live here. No magic numbers in code.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Zero-Trust Banking MFA"
    app_version: str = "0.1.0"
    debug: bool = True

    # Server
    host: str = "127.0.0.1"
    port: int = 8000

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # LM Studio (local LLM server)
    llm_base_url: str = "http://localhost:1234/v1"
    llm_model: str = "qwen2.5-14b-instruct"
    llm_timeout_seconds: float = 30.0
    llm_max_tokens: int = 512
    llm_temperature: float = 0.3

    # Database
    database_url: str = "sqlite:///./zerotrust.db"

    # Security
    jwt_secret: str = "CHANGE_ME_IN_ENV_FILE"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_days: int = 7

    # Biometric
    biometric_match_threshold: float = 0.3
    biometric_reverify_interval_seconds: int = 30

    # Risk Engine
    risk_threshold_low: float = 0.3
    risk_threshold_high: float = 0.7

    # Audit
    audit_log_path: str = "./audit_logs/chain.jsonl"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance. Use this everywhere instead of instantiating Settings()."""
    return Settings()

