"""Typed application settings loaded from the environment.

No hardcoded hyperparameters - everything configurable lives here.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "development"
    log_level: str = "INFO"
    seed: int = 42

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    wandb_project: str = "ml-graph-compiler"


settings = Settings()
