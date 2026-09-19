"""Minimal valid predictor; replace this abstaining implementation."""

from __future__ import annotations

from typing import Any


def predict(request: dict[str, Any]) -> dict[str, str]:
    """Return the canonical explicit-abstention response."""
    del request
    return {"slug": "not_found"}
