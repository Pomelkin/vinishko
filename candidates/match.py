"""Deterministic Catalog candidate pool from extracted image attributes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from candidates.catalog import UNKNOWN, catalog_rows, catalog_values


def match_products(prediction: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return all Catalog cards matching the known fields, without arbitrary ranking.

    An unknown field places no restriction. With two unknown fields there is no
    evidence to form a useful product pool, so the result is empty.
    """
    if prediction.get("_status", "ok") != "ok":
        return []
    category = prediction.get("category")
    brand = prediction.get("brand")
    categories, brands = catalog_values()
    if category not in (*categories, UNKNOWN) or brand not in (*brands, UNKNOWN):
        raise ValueError("Prediction contains a value outside the Catalog")
    if category == brand == UNKNOWN:
        return []
    return [
        dict(row)
        for row in catalog_rows()
        if (category == UNKNOWN or row["Категория"].strip() == category)
        and (brand == UNKNOWN or row["Винодельня"].strip() == brand)
    ]
