"""Configuration for the stateless OpenRouter sommelier call."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import StringConstraints


NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
ReasoningEffort = Literal[
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
] | Annotated[int, Field(ge=1, le=100)]

PACKAGE_DIR = Path(__file__).resolve().parent
AI_SOMELIER_DIR = PACKAGE_DIR.parent.parent
DEFAULT_KNOWLEDGE_PATH = (
    AI_SOMELIER_DIR / "kb" / "EXPERT_KNOWLEDGE.json"
)


class StrictSettings(BaseModel):
    """Reject undeclared settings and accidental type coercion."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class GenerationSettings(StrictSettings):
    """Text-generation controls shared by first and later turns."""

    temperature: float = Field(default=0.2, ge=0, le=2)
    max_completion_tokens: int = Field(default=32768, ge=1)
    reasoning_effort: ReasoningEffort | None = "minimal"
    reasoning_exclude: bool = False

    def request_payload(self) -> dict[str, Any]:
        """Serialize OpenRouter-compatible generation parameters."""
        payload: dict[str, Any] = {
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
        }
        if self.reasoning_effort is not None:
            payload["reasoning"] = {
                "effort": self.reasoning_effort,
                "exclude": self.reasoning_exclude,
            }
        return payload


class ProviderSettings(StrictSettings):
    """OpenRouter routing controls matching the repository's NDR calls."""

    only: tuple[NonEmptyString, ...] = ("together",)
    allow_fallbacks: bool = False
    require_parameters: bool = True

    def request_payload(self) -> dict[str, Any]:
        """Serialize provider routing without empty values."""
        payload = self.model_dump(mode="json")
        if not payload["only"]:
            payload.pop("only")
        return payload


class SommelierSettings(StrictSettings):
    """Complete model, connection, prompt, and knowledge configuration."""

    model: NonEmptyString = "deepseek/deepseek-v4.1-flash"
    api_base: NonEmptyString = "https://openrouter.ai/api/v1"
    api_key_env: NonEmptyString = "OPENROUTER_API_KEY"
    http_referer_env: NonEmptyString = "OPENROUTER_SITE_URL"
    app_title_env: NonEmptyString = "OPENROUTER_APP_TITLE"
    user_agent: NonEmptyString = "vinishko-ai-sommelier/2"
    timeout_seconds: float = Field(default=30.0, gt=0, le=3600)
    prompt_version: NonEmptyString = "v2"
    expert_knowledge_path: Path = DEFAULT_KNOWLEDGE_PATH
    generation: GenerationSettings = GenerationSettings()
    provider: ProviderSettings = ProviderSettings()


SETTINGS = SommelierSettings()


def settings_with_overrides(
    overrides: Any,
    base: SommelierSettings = SETTINGS,
) -> SommelierSettings:
    """Apply an optional runner-owned runtime object to default settings."""
    if overrides is None:
        return base
    if not isinstance(overrides, dict):
        raise TypeError("request.runtime must be an object")
    merged = base.model_dump()
    for key, value in overrides.items():
        runtime_value = (
            Path(value)
            if key == "expert_knowledge_path" and isinstance(value, str)
            else value
        )
        if key in {"generation", "provider"} and isinstance(runtime_value, dict):
            nested = dict(merged[key])
            nested.update(runtime_value)
            merged[key] = nested
        else:
            merged[key] = runtime_value
    return SommelierSettings.model_validate(merged)
