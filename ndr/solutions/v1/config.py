"""Single editable configuration for NDR model calls and evaluation runs."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from pydantic import model_validator


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

    routing_mode: RoutingMode | None = None
    order: tuple[NonEmptyString, ...] = ()
    only: tuple[NonEmptyString, ...] = ()
    ignore: tuple[NonEmptyString, ...] = ()
    allow_fallbacks: bool | None = None
    require_parameters: bool | None = True
    data_collection: Literal["allow", "deny"] | None = None
    zdr: bool | None = None
    quantizations: tuple[NonEmptyString, ...] = ()
    max_price: ProviderPriceLimits | None = None

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

    model: NonEmptyString | None = None
    api_base: NonEmptyString = "https://openrouter.ai/api/v1"
    api_key_env: NonEmptyString = "OPENROUTER_API_KEY"
    http_referer_env: NonEmptyString = "OPENROUTER_SITE_URL"
    app_title_env: NonEmptyString = "OPENROUTER_APP_TITLE"
    user_agent: NonEmptyString = "vinishko-ndr-runner/1"
    routing: ProviderRoutingSettings = ProviderRoutingSettings()


class GenerationSettings(StrictSettings):
    """Parameters applied identically to every comparison and resolver call."""

    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    min_p: float | None = Field(default=None, ge=0, le=1)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    repetition_penalty: float | None = Field(default=None, gt=0, le=2)
    max_completion_tokens: int = Field(default=16000, ge=1)
    generations: int = Field(default=1, ge=1, le=16)
    seed: int | None = None
    stop: tuple[NonEmptyString, ...] = Field(default=(), max_length=4)
    image_detail: ImageDetail = "original"
    reasoning_effort: ReasoningEffort | None = "minimal"
    reasoning_exclude: bool = False

    def request_payload(self) -> dict[str, Any]:
        """Serialize names expected by the OpenAI-compatible request body."""
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
        if self.reasoning_effort is not None:
            payload["reasoning"] = {
                "effort": self.reasoning_effort,
                "exclude": self.reasoning_exclude,
            }
        return payload


class ExecutionSettings(StrictSettings):
    """Controls for the local evaluation runner rather than model sampling."""

    timeout_seconds: float = Field(default=120.0, gt=0, le=3600)
    concurrency: int = Field(default=1, ge=1, le=64)
    candidate_order_seed: int = 0
    limit: int | None = Field(default=None, ge=1)


class NdrSettings(StrictSettings):
    """Complete editable configuration for the NDR solution."""

    openrouter: OpenRouterSettings
    generation: GenerationSettings
    execution: ExecutionSettings


# Edit this block. CLI flags and OPENROUTER_* environment variables may override
# connection/run values, but provider routing lives here as the single source.
SETTINGS = NdrSettings(
    openrouter=OpenRouterSettings(
        model=None,
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        http_referer_env="OPENROUTER_SITE_URL",
        app_title_env="OPENROUTER_APP_TITLE",
        user_agent="vinishko-ndr-runner/1",
        routing=ProviderRoutingSettings(
            routing_mode=None,  # Nitro: fastest generation throughput
            order=(),
            only=("together",),
            ignore=(),
            allow_fallbacks=False,
            require_parameters=True,
            data_collection=None,
            zdr=None,
            quantizations=(),
            max_price=None,
        ),
    ),
    generation=GenerationSettings(
        temperature=0.0,
        top_p=None,
        top_k=None,
        min_p=None,
        frequency_penalty=None,
        presence_penalty=None,
        repetition_penalty=None,
        max_completion_tokens=4096,
        generations=1,
        seed=None,
        stop=(),
        image_detail="original",
        reasoning_effort="minimal",
        reasoning_exclude=False,
    ),
    execution=ExecutionSettings(
        timeout_seconds=120.0,
        concurrency=1,
        candidate_order_seed=0,
        limit=None,
    ),
)
