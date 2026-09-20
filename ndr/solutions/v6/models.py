"""Strict Pydantic contracts and deterministic decisions for NDR v6."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic import create_model


EvidenceState = Literal["match", "conflict", "unknown"]
Verdict = Literal["same", "different"]

DISCRIMINATORS = (
    "maker",
    "product_name",
    "line_variant",
    "grape",
    "color",
    "sweetness",
    "carbonation",
    "vintage",
)
IDENTITY_DISCRIMINATORS = ("product_name", "line_variant")
STYLE_DISCRIMINATORS = ("color", "sweetness", "carbonation")


class ModelContractError(ValueError):
    """Raised when a decoded generation violates its stage contract."""


class StrictOutput(BaseModel):
    """Forbid coercion and undeclared fields in all model outputs."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


class EvidenceItem(StrictOutput):
    """Observed values and their three-state comparison."""

    query_value: str = Field(
        min_length=1,
        max_length=160,
        description=(
            "Concise value observed on QUERY, or a short reason it is unavailable."
        ),
    )
    element_value: str = Field(
        min_length=1,
        max_length=160,
        description=(
            "Concise value observed in the ELEMENT reference/card, or a short reason "
            "it is unavailable."
        ),
    )
    state: EvidenceState = Field(
        description=(
            "match when reliable values agree, conflict when reliable values disagree, "
            "unknown when either side lacks reliable evidence."
        ),
    )


class ComparisonChecklist(StrictOutput):
    """Eight atomic SKU discriminators required for every comparison."""

    maker: EvidenceItem
    product_name: EvidenceItem
    line_variant: EvidenceItem
    grape: EvidenceItem
    color: EvidenceItem
    sweetness: EvidenceItem
    carbonation: EvidenceItem
    vintage: EvidenceItem


class YearMattersOutput(StrictOutput):
    """Evidence for an ELEMENT whose exact catalog vintage identifies the SKU."""

    checklist: ComparisonChecklist


class YearNotMatterOutput(StrictOutput):
    """Evidence for an ELEMENT whose vintage is excluded from the SKU decision."""

    checklist: ComparisonChecklist


def comparison_output_model(year_matters: bool) -> type[BaseModel]:
    """Select the only valid evidence contract for the current ELEMENT."""
    return YearMattersOutput if year_matters else YearNotMatterOutput


def comparison_verdict(
    payload: Mapping[str, Any],
    *,
    year_matters: bool,
) -> Verdict:
    """Derive the pairwise verdict from validated atomic evidence.

    Any relevant explicit conflict rejects the candidate. Unknown evidence is never a
    conflict, but a positive decision still needs a product identity anchor. When the
    catalog defines a vintage, an exact vintage match is mandatory. When it does not,
    vintage evidence remains observable but cannot affect the decision.
    """
    checklist = payload.get("checklist")
    if not isinstance(checklist, Mapping):
        raise ModelContractError("comparison payload has no checklist object")

    states: dict[str, str] = {}
    for name in DISCRIMINATORS:
        item = checklist.get(name)
        if not isinstance(item, Mapping):
            raise ModelContractError(f"checklist.{name} must be an object")
        state = item.get("state")
        if state not in ("match", "conflict", "unknown"):
            raise ModelContractError(f"checklist.{name}.state is invalid")
        states[name] = state

    relevant = DISCRIMINATORS if year_matters else DISCRIMINATORS[:-1]
    if any(states[name] == "conflict" for name in relevant):
        return "different"
    if year_matters and states["vintage"] != "match":
        return "different"

    matched = {name for name in relevant if states[name] == "match"}
    if matched.intersection(IDENTITY_DISCRIMINATORS) and len(matched) >= 2:
        return "same"

    maker_grape_style = (
        states["maker"] == "match"
        and states["grape"] == "match"
        and any(states[name] == "match" for name in STYLE_DISCRIMINATORS)
    )
    return "same" if maker_grape_style else "different"


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
