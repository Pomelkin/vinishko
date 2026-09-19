"""Strict Pydantic contracts for every NDR model stage."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError
from pydantic import create_model
from pydantic import model_validator


Verdict = Literal["same", "different"]


class ModelContractError(ValueError):
    """Raised when a decoded generation violates its stage contract."""


class StrictOutput(BaseModel):
    """Forbid coercion and undeclared fields in all model outputs."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


class ChecklistItem(StrictOutput):
    """One concise observed fact and its binary comparison result."""

    observation: str = Field(min_length=1, max_length=240)
    is_same: StrictBool


class YearMattersChecklist(StrictOutput):
    """Required evidence when the catalog element has a vintage."""

    maker: ChecklistItem
    profile: ChecklistItem
    year: ChecklistItem


class YearNotMatterChecklist(StrictOutput):
    """Required evidence when the catalog element has no vintage."""

    maker: ChecklistItem
    profile: ChecklistItem


class YearMattersOutput(StrictOutput):
    """Comparison output whose verdict includes exact vintage agreement."""

    checklist: YearMattersChecklist
    verdict: Verdict

    @model_validator(mode="after")
    def verdict_matches_checklist(self) -> YearMattersOutput:
        """Reject a verdict inconsistent with any required evidence flag."""
        flags = (
            self.checklist.maker.is_same,
            self.checklist.profile.is_same,
            self.checklist.year.is_same,
        )
        if (self.verdict == "same") != all(flags):
            raise ValueError("verdict contradicts checklist flags")
        return self


class YearNotMatterOutput(StrictOutput):
    """Comparison output that deliberately excludes vintage agreement."""

    checklist: YearNotMatterChecklist
    verdict: Verdict

    @model_validator(mode="after")
    def verdict_matches_checklist(self) -> YearNotMatterOutput:
        """Reject a verdict inconsistent with either required evidence flag."""
        flags = (
            self.checklist.maker.is_same,
            self.checklist.profile.is_same,
        )
        if (self.verdict == "same") != all(flags):
            raise ValueError("verdict contradicts checklist flags")
        return self


def comparison_output_model(year_matters: bool) -> type[BaseModel]:
    """Select the only valid comparison contract for the current element."""
    return YearMattersOutput if year_matters else YearNotMatterOutput


def resolver_output_model(allowed_slugs: Iterable[str]) -> type[BaseModel]:
    """Build a one-field model whose slug is limited to current matches."""
    slugs = tuple(allowed_slugs)
    if not slugs:
        raise ModelContractError("resolver requires at least one allowed slug")
    if any(not isinstance(slug, str) or not slug for slug in slugs):
        raise ModelContractError("every resolver slug must be a non-empty string")
    if len(set(slugs)) != len(slugs):
        raise ModelContractError("resolver slugs must be unique")

    slug_literal = Literal.__getitem__(slugs)
    return create_model(
        "ResolverOutput",
        __base__=StrictOutput,
        slug=(slug_literal, ...),
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
