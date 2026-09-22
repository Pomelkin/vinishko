"""Strict contracts for the sommelier's first structured response."""

from __future__ import annotations

import re
from typing import Annotated
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import StringConstraints
from pydantic import ValidationError
from pydantic import model_validator


FIRST_TURN_CLOSING = "Чем я могу вам помочь?"
Suggestion = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=29),
]


class OutputContractError(ValueError):
    """Raised when a model generation violates its turn contract."""


class FirstTurnOutput(BaseModel):
    """Mobile-sized opening answer and exactly two generative suggestions."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    content: str = Field(min_length=350, max_length=750)
    suggestions: list[Suggestion] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def enforce_copy_contract(self) -> FirstTurnOutput:
        """Check requirements that JSON Schema cannot express cleanly."""
        if len(re.findall(r"\S+", self.content)) > 100:
            raise ValueError("content must contain at most 100 words")
        if self.content.count("?") != 1:
            raise ValueError("content must contain exactly one question mark")
        if not self.content.endswith(FIRST_TURN_CLOSING):
            raise ValueError(
                f"content must end with {FIRST_TURN_CLOSING!r}",
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

