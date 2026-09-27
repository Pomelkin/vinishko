"""Strict Pydantic contract for one-shot candidate-or-not-found selection."""

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
    """Static portion of the one-shot output contract."""

    checklist: SelectionChecklist


def nearest_output_model(allowed_slugs: Iterable[str]) -> type[BaseModel]:
    """Build a strict model allowing a group slug or an explicit no-match result."""
    slugs = tuple(allowed_slugs)
    if not slugs:
        raise ModelContractError("nearest selection requires at least one allowed slug")
    if any(not isinstance(slug, str) or not slug for slug in slugs):
        raise ModelContractError("every allowed slug must be a non-empty string")
    if len(set(slugs)) != len(slugs):
        raise ModelContractError("allowed slugs must be unique")
    if NOT_FOUND in slugs:
        raise ModelContractError("not_found is reserved and cannot be a candidate slug")

    slug_literal = Literal.__getitem__((*slugs, NOT_FOUND))
    return create_model(
        "NearestOutput",
        __base__=NearestOutputBase,
        slug=(slug_literal, ...),
    )