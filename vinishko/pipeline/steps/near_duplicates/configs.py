"""Параметры вызова NDR v5 внутри pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from typing import Any
from typing import Literal

import yaml
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import StringConstraints
from pydantic import field_validator
from pydantic import model_validator


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"


NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
RoutingMode = Literal["price", "throughput", "latency"]
ImageDetail = Literal["auto", "low", "high", "original"]
ReasoningEffortName = Literal[
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
]
ReasoningEffort = ReasoningEffortName | Annotated[int, Field(ge=1, le=100)]


class StrictSettings(BaseModel):
    """Reject unknown keys and accidental type coercion in local settings."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ProviderPriceLimits(StrictSettings):
    """Optional OpenRouter price ceilings in provider API units."""

    prompt: float | None = Field(default=None, ge=0)
    completion: float | None = Field(default=None, ge=0)
    image: float | None = Field(default=None, ge=0)
    request: float | None = Field(default=None, ge=0)


class ProviderRoutingSettings(StrictSettings):
    """OpenRouter provider preferences serialized into ``payload.provider``."""

    routing_mode: RoutingMode | None
    order: tuple[NonEmptyString, ...]
    only: tuple[NonEmptyString, ...]
    ignore: tuple[NonEmptyString, ...]
    allow_fallbacks: bool | None
    require_parameters: bool | None
    data_collection: Literal["allow", "deny"] | None
    zdr: bool | None
    quantizations: tuple[NonEmptyString, ...]
    max_price: ProviderPriceLimits | None

    @field_validator("order", "only", "ignore", "quantizations", mode="before")
    @classmethod
    def yaml_list_to_tuple(cls, value: Any) -> Any:
        """YAML sequences are lists, while settings expose immutable tuples."""
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def provider_lists_are_consistent(self) -> ProviderRoutingSettings:
        """Reject ambiguous or duplicated provider filters."""
        for field_name in ("order", "only", "ignore", "quantizations"):
            values = getattr(self, field_name)
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} contains duplicates")
        if set(self.only) & set(self.ignore):
            raise ValueError("the same provider cannot be present in only and ignore")
        return self

    def request_payload(self) -> dict[str, Any]:
        """Return only explicitly useful OpenRouter provider preferences."""
        payload = self.model_dump(mode="json", exclude_none=True)
        routing_mode = payload.pop("routing_mode", None)
        if routing_mode is not None:
            payload["sort"] = routing_mode
        for key in ("order", "only", "ignore", "quantizations"):
            if not payload.get(key):
                payload.pop(key, None)
        if not payload.get("max_price"):
            payload.pop("max_price", None)
        return payload


class OpenRouterSettings(StrictSettings):
    """Connection, identity headers, and provider routing preferences."""

    model: NonEmptyString | None
    api_base: NonEmptyString
    api_key_env: NonEmptyString
    http_referer_env: NonEmptyString
    app_title_env: NonEmptyString
    user_agent: NonEmptyString
    routing: ProviderRoutingSettings


class ReasoningByTask(StrictSettings):
    """Reasoning effort per call task; None omits the reasoning parameter."""

    group: ReasoningEffort | None
    final: ReasoningEffort | None


class GenerationSettings(StrictSettings):
    """Parameters applied identically to every comparison and resolver call."""

    temperature: float = Field(ge=0, le=2)
    top_p: float | None = Field(gt=0, le=1)
    top_k: int | None = Field(ge=0)
    min_p: float | None = Field(ge=0, le=1)
    frequency_penalty: float | None = Field(ge=-2, le=2)
    presence_penalty: float | None = Field(ge=-2, le=2)
    repetition_penalty: float | None = Field(gt=0, le=2)
    max_completion_tokens: int = Field(ge=1)
    generations: int = Field(ge=1, le=16)
    seed: int | None
    stop: tuple[NonEmptyString, ...] = Field(max_length=4)
    image_detail: ImageDetail
    reasoning_effort: ReasoningByTask
    """How much the model reasons before answering, separately for the group round and the final."""
    reasoning_exclude: bool

    @field_validator("stop", mode="before")
    @classmethod
    def yaml_list_to_tuple(cls, value: Any) -> Any:
        """Keep the strict tuple field compatible with YAML sequences."""
        return tuple(value) if isinstance(value, list) else value

    def request_payload(self, task: Literal["group", "final"]) -> dict[str, Any]:
        """Serialize names expected by the OpenAI-compatible request body for one call task."""
        payload = self.model_dump(
            mode="json",
            exclude={
                "generations",
                "image_detail",
                "reasoning_effort",
                "reasoning_exclude",
            },
            exclude_none=True,
        )
        if self.generations != 1:
            payload["n"] = self.generations
        if not payload.get("stop"):
            payload.pop("stop", None)
        effort = getattr(self.reasoning_effort, task)
        if effort is not None:
            payload["reasoning"] = {
                "effort": effort,
                "exclude": self.reasoning_exclude,
            }
        return payload


class ExecutionSettings(StrictSettings):
    """Execution limits rather than model sampling."""

    timeout_seconds: float = Field(gt=0, le=3600)
    """Wall-clock limit of one model call including its retries."""
    retries: int = Field(ge=0, le=10)
    """Extra attempts after HTTP 429 and 500/502/503/504 with exponential backoff and jitter."""
    concurrency: int = Field(ge=1, le=64)
    """Simultaneous model calls per resolver; copies made by with_trace_dir share the limit."""
    candidate_order_seed: int
    limit: int | None = Field(ge=1)


class NdrSettings(StrictSettings):
    """Complete editable configuration for the NDR solution."""

    openrouter: OpenRouterSettings
    generation: GenerationSettings
    execution: ExecutionSettings


def load_config(path: Path | None = None) -> NdrSettings:
    """Read and validate NDR settings from YAML."""
    path = (path or DEFAULT_CONFIG).resolve()
    return NdrSettings.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
