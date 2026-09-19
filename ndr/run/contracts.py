"""Public contract between the near-duplicate runner and a predictor."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SCHEMA_VERSION = 1
NOT_FOUND = "not_found"


class ContractError(ValueError):
    """Raised when a predictor response violates the runner contract."""


def prediction_slug(response: Any, allowed_slugs: set[str]) -> str:
    """Validate a predictor response and return its canonical slug."""
    if not isinstance(response, Mapping):
        raise ContractError('predict() must return an object like {"slug": "..."}')

    slug = response.get("slug")
    if slug == "not found":
        slug = NOT_FOUND
    if not isinstance(slug, str) or not slug:
        raise ContractError('response field "slug" must be a non-empty string')
    if slug != NOT_FOUND and slug not in allowed_slugs:
        raise ContractError(
            f'predicted slug is outside the offered group: "{slug}"',
        )
    return slug
