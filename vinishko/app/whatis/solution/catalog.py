"""The bundled Catalog values accepted by the copied solution."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


UNKNOWN = "Не удалось определить"
VALUES_PATH = Path(__file__).resolve().parent.parent / "kb" / "catalog_values.json"


@lru_cache(maxsize=1)
def catalog_values() -> tuple[tuple[str, ...], tuple[str, ...]]:
    values = json.loads(VALUES_PATH.read_text(encoding="utf-8"))
    categories = tuple(values["categories"])
    brands = tuple(values["brands"])
    if not categories or not brands or any(not value for value in (*categories, *brands)):
        raise ValueError("Каталог содержит пустую категорию или винодельню")
    return categories, brands
