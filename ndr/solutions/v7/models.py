"""Strict Pydantic contract for image-only candidate selection."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic import create_model


NOT_FOUND = "not_found"


class ModelContractError(ValueError):
    """Raised when a decoded generation violates its stage contract."""


class StrictOutput(BaseModel):
    """Forbid coercion and undeclared fields in all model outputs."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


class SelectionObservation(StrictOutput):
    """One concise observation supporting the final decision."""

    observation: str = Field(min_length=1, max_length=240)


class SelectionChecklist(StrictOutput):
    """Required evidence for the candidate or no-match decision."""

    maker: SelectionObservation
    profile: SelectionObservation
    year: SelectionObservation


class NearestOutputBase(StrictOutput):
    """Static portion of the image-only one-shot output contract."""

    checklist: SelectionChecklist


def nearest_output_model(allowed_elements: Iterable[str]) -> type[BaseModel]:
    """Build a strict model allowing an opaque element id or no-match result."""
    elements = tuple(allowed_elements)
    if not elements:
        raise ModelContractError("nearest selection requires at least one element")
    if any(not isinstance(element, str) or not element for element in elements):
        raise ModelContractError("every allowed element must be a non-empty string")
    if len(set(elements)) != len(elements):
        raise ModelContractError("allowed elements must be unique")
    if NOT_FOUND in elements:
        raise ModelContractError("not_found is reserved and cannot be an element id")

    selection_literal = Literal.__getitem__((*elements, NOT_FOUND))
    return create_model(
        "NearestOutput",
        __base__=NearestOutputBase,
        selection=(selection_literal, ...),
    )


def response_format(name: str, output_model: type[BaseModel]) -> dict[str, Any]:
    """Build an OpenAI-compatible strict structured-output declaration."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": output_model.model_json_schema(mode="validation"),
        },
    }


def validate_output(payload: Any, output_model: type[BaseModel]) -> dict[str, Any]:
    """Validate a decoded response with the same model sent to the provider."""
    try:
        validated = output_model.model_validate(payload)
    except ValidationError as error:
        raise ModelContractError(str(error)) from error
    return validated.model_dump(mode="json")
