"""Strict contracts for the sommelier's first structured response."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError
from pydantic import model_validator


FIRST_TURN_CLOSING = "Чем я могу помочь?"


class OutputContractError(ValueError):
    """Raised when a model generation violates its turn contract."""


class FirstTurnOutput(BaseModel):
    """Mobile-sized opening answer and exactly two generative suggestions."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    content: str
    suggestions: list[str] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def enforce_copy_contract(self) -> FirstTurnOutput:
        """Check requirements that JSON Schema cannot express cleanly."""
        if not self.content.strip():
            raise ValueError("content must not be empty")
        self.suggestions = [
            suggestion.rstrip(" \t\r\n?？") for suggestion in self.suggestions
        ]
        if any(not suggestion for suggestion in self.suggestions):
            raise ValueError("suggestions must not be empty")
        if not self.content.endswith("\n" + FIRST_TURN_CLOSING):
            raise ValueError(
                f"content must end with {FIRST_TURN_CLOSING!r} on a separate line",
            )
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
    """Validate a decoded first generation with the provider-visible model."""
    try:
        validated = FirstTurnOutput.model_validate(payload)
    except ValidationError as error:
        raise OutputContractError(str(error)) from error
    return validated.model_dump(mode="json")
