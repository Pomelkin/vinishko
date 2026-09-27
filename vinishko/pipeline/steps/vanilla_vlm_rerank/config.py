"""Настройки отдельного vanilla VLM шага."""

from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class VanillaVlmSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[4] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    model: str = Field(default="deepseek/deepseek-v4.1-flash", validation_alias="OPENROUTER_MODEL")
    api_base: str = Field(default="https://openrouter.ai/api/v1", validation_alias="OPENROUTER_API_BASE")
    http_proxy: str | None = Field(default=None, validation_alias="openrouter_http_proxy")
    site_url: str | None = Field(default=None, validation_alias="OPENROUTER_SITE_URL")
    app_title: str | None = Field(default=None, validation_alias="OPENROUTER_APP_TITLE")
    timeout_seconds: float = Field(default=60.0, gt=0, validation_alias="VANILLA_VLM_TIMEOUT_SECONDS")
    max_completion_tokens: int = Field(default=4096, gt=0, validation_alias="VANILLA_VLM_MAX_COMPLETION_TOKENS")
    extra_body: dict[str, Any] = Field(
        default_factory=lambda: {
            "reasoning": {"enabled": False},
            "provider": {"only": ["together"]}
        },
        validation_alias="VANILLA_VLM_EXTRA_BODY",
    )
