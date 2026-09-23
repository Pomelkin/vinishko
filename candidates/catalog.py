"""The Catalog is the sole source of allowed category and winery values."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "data" / "strapi" / "catalog_dataset.csv"
UNKNOWN = "Не удалось определить"


@lru_cache(maxsize=1)
def catalog_values() -> tuple[tuple[str, ...], tuple[str, ...]]:
    rows = catalog_rows()
    if not rows:
        raise ValueError("Каталог пуст")
    categories = tuple(sorted({row["Категория"].strip() for row in rows}))
    brands = tuple(sorted({row["Винодельня"].strip() for row in rows}))
    if not categories or not brands or "" in categories or "" in brands:
        raise ValueError("Каталог содержит пустую категорию или винодельню")
    return categories, brands


@lru_cache(maxsize=1)
def catalog_rows() -> tuple[dict[str, str], ...]:
    with CATALOG_PATH.open(encoding="utf-8-sig", newline="") as source:
        return tuple(csv.DictReader(source))
