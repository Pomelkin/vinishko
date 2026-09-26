"""Strict contracts for the sommelier's first structured response."""

from __future__ import annotations

from typing import Any
from typing import Mapping

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import ValidationError
from pydantic import model_validator


class OutputContractError(ValueError):
    """Raised when a model generation violates its turn contract."""


class FirstTurnOutput(BaseModel):
    """Structured opening answer from the provider."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    content: str
    suggestions: list[str] | None

    @model_validator(mode="after")
    def ensure_content(self) -> FirstTurnOutput:
        """Reject an empty assistant message."""
        if not self.content.strip():
            raise ValueError("content must not be empty")
        return self


def response_format() -> dict[str, Any]:
    """Return the strict OpenAI-compatible first-turn response schema."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "sommelier_first_turn",
            "strict": True,
            "schema": FirstTurnOutput.model_json_schema(mode="validation"),
        },
    }


def validate_first_turn(payload: Any) -> dict[str, Any]:
    """Validate the message and keep usable suggestions without judging their copy."""
    if isinstance(payload, Mapping):
        suggestions = payload.get("suggestions")
        payload = {
            "content": payload.get("content"),
            "suggestions": suggestions
            if isinstance(suggestions, list) and all(isinstance(item, str) for item in suggestions)
            else None,
        }
    try:
        validated = FirstTurnOutput.model_validate(payload)
    except ValidationError as error:
        raise OutputContractError(str(error)) from error
    return validated.model_dump(mode="json")
