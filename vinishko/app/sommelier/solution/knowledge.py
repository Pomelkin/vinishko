"""Select expert blocks from recognized-wine fields only."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


NO_DATA = "Нет данных"
MAX_EXPERT_VALUE_CHARACTERS = 500
TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "EXPERT_KNOWLEDGE_TEMPLATE.json"
)

PROMPT_CARD_FIELDS = (
    "Название вина",
    "Категория",
    "Цвет",
    "Регион",
    "Сорт винограда",
    "Описание",
    "Винодельня",
    "Slug",
    "Винтаж",
    "Крепость, %",
    "Крепость от, %",
    "Крепость до, %",
    "Сахар",
    "Игристое",
    "Метод игристого",
    "Креплёное",
    "Безалкогольное",
    "Органическое",
    "Выдержка или резерв",
)

SWEETNESS_VALUES = {
    "брют натюр": "brut_nature",
    "экстра брют": "extra_brut",
    "брют": "brut",
    "сухое": "dry",
    "полусухое": "semi_dry",
    "полусладкое": "semi_sweet",
    "сладкое": "sweet",
    "десертное": "dessert",
}
COLOR_VALUES = {
    "белое": "white",
    "красное": "red",
    "розовое": "rose",
    "оранжевое": "orange",
}
SPARKLING_METHOD_VALUES = {
    "петнат": "petnat",
    "классический": "traditional",
}


class KnowledgeFormatError(ValueError):
    """Raised when a filled expert file does not match the strict template."""


def is_missing(value: Any) -> bool:
    """Recognize empty CSV values, Python nulls, and numeric NaN values."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def display_value(value: Any) -> str:
    """Render every absent card value explicitly for the language model."""
    if is_missing(value):
        return NO_DATA
    return str(value).strip()


def normalize_catalog_card(card: Mapping[str, Any]) -> dict[str, str]:
    """Keep prompt-relevant Catalog columns and make every unknown visible."""
    if not isinstance(card, Mapping):
        raise TypeError("a catalog card must be an object")
    return {field: display_value(card.get(field)) for field in PROMPT_CARD_FIELDS}


def load_expert_knowledge(path: Path) -> dict[str, dict[str, list[str]]]:
    """Load and strictly validate a filled expert-knowledge JSON file."""
    template = _load_json_object(TEMPLATE_PATH)
    knowledge = _load_json_object(path)
    if set(knowledge) != set(template):
        raise KnowledgeFormatError(
            "expert knowledge axes must exactly match EXPERT_KNOWLEDGE_TEMPLATE.json",
        )

    validated: dict[str, dict[str, list[str]]] = {}
    for axis, template_values in template.items():
        values = knowledge[axis]
        if not isinstance(values, dict) or set(values) != set(template_values):
            raise KnowledgeFormatError(
                f"expert knowledge values for {axis!r} must exactly match the template",
            )
        validated[axis] = {}
        for value, bullets in values.items():
            if not isinstance(bullets, list) or any(
                not isinstance(bullet, str) or not bullet.strip()
                for bullet in bullets
            ):
                raise KnowledgeFormatError(
                    f"expert knowledge {axis}.{value} must be a list of non-empty strings",
                )
            cleaned = [bullet.strip() for bullet in bullets]
            if sum(len(bullet) for bullet in cleaned) > MAX_EXPERT_VALUE_CHARACTERS:
                raise KnowledgeFormatError(
                    f"expert knowledge {axis}.{value} exceeds 500 characters",
                )
            validated[axis][value] = cleaned
    return validated


def _load_json_object(path: Path) -> dict[str, Any]:
    """Decode a UTF-8 JSON object from a concrete path."""
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise KnowledgeFormatError(f"{path} must contain a JSON object")
    return payload


def selected_knowledge_values(card: Mapping[str, Any]) -> dict[str, str]:
    """Derive one template value per applicable axis from one recognized wine."""
    category = _normalized(card.get("Категория"))
    sugar = _normalized(card.get("Сахар"))
    method = _normalized(card.get("Метод игристого"))
    grape = _text(card.get("Сорт винограда"))
    region = _text(card.get("Регион"))
    brand = _text(card.get("Винодельня"))

    return {
        "color": COLOR_VALUES.get(category, "unknown"),
        "effervescence": _effervescence(card),
        "sweetness": SWEETNESS_VALUES.get(sugar, "unknown"),
        "sparkling_method": SPARKLING_METHOD_VALUES.get(method, "unknown"),
        "special_style": _special_style(card),
        "fortified": "confirmed" if _confirmed(card.get("Креплёное")) else "unknown",
        "alcohol_band": _alcohol_band(card),
        "grape_composition": _grape_composition(grape),
        "grape": grape or "unknown",
        "region": region or "unknown",
        "brand": brand or "unknown",
        "aging_or_reserve": (
            "confirmed"
            if _confirmed(card.get("Выдержка или резерв"))
            else "unknown"
        ),
        "organic": "confirmed" if _confirmed(card.get("Органическое")) else "unknown",
    }


def compile_expert_knowledge(
    card: Mapping[str, Any],
    knowledge: Mapping[str, Mapping[str, list[str]]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Return selected values and only their non-empty expert blocks."""
    selections = selected_knowledge_values(card)
    blocks: list[dict[str, Any]] = []
    for axis, requested_value in selections.items():
        axis_values = knowledge.get(axis, {})
        selected_value = requested_value
        if selected_value not in axis_values:
            if "unknown" not in axis_values:
                continue
            selected_value = "unknown"
        selections[axis] = selected_value
        bullets = axis_values[selected_value]
        if bullets:
            blocks.append(
                {
                    "axis": axis,
                    "value": selected_value,
                    "bullets": list(bullets),
                },
            )
    return selections, blocks


def _text(value: Any) -> str:
    """Return a trimmed scalar or an empty string for missing data."""
    return "" if is_missing(value) else str(value).strip()


def _normalized(value: Any) -> str:
    """Normalize a scalar for exact categorical matching."""
    return _text(value).casefold().replace("ё", "е")


def _confirmed(value: Any) -> bool:
    """Treat only an explicit affirmative value as confirmed."""
    return _normalized(value) == "да"


def _effervescence(card: Mapping[str, Any]) -> str:
    """Distinguish confirmed sparkling and frizzante from unknown state."""
    searchable = " ".join(
        _normalized(card.get(field))
        for field in ("Название вина", "Описание")
    )
    if re.search(r"\b(frizzante|фриззанте)\b", searchable):
        return "semi_sparkling"
    if _confirmed(card.get("Игристое")):
        return "sparkling"
    return "unknown"


def _special_style(card: Mapping[str, Any]) -> str:
    """Select the most specific special style explicitly supported by the card."""
    searchable = " ".join(
        _normalized(card.get(field))
        for field in (
            "Название вина",
            "Категория",
            "Описание",
            "Метод игристого",
        )
    )
    patterns = (
        ("petnat", r"\b(пет[ -]?нат|pet[ -]?nat|petnat)\b"),
        ("frizzante", r"\b(frizzante|фриззанте)\b"),
        ("spumante", r"\b(spumante|спуманте)\b"),
        ("ice_wine", r"\b(ice[ -]?wine|айсвайн|ледяное вино)\b"),
        ("port_style", r"\b(портвейн|порто)\b"),
        ("madeira_style", r"\b(мадера|madeira)\b"),
        ("sherry_style", r"\b(херес|sherry)\b"),
        ("kagor", r"\b(кагор|cahors)\b"),
        ("orange", r"\b(оранжевое|оранж|orange)\b"),
    )
    for value, pattern in patterns:
        if re.search(pattern, searchable):
            return value
    if _confirmed(card.get("Креплёное")):
        return "fortified_other"
    return "unknown"


def _number(value: Any) -> float | None:
    """Extract one decimal number from a Catalog strength field."""
    text = _text(value).replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def _alcohol_band(card: Mapping[str, Any]) -> str:
    """Map exact strength or a range midpoint to the knowledge bands."""
    alcohol = _number(card.get("Крепость, %"))
    if alcohol is None:
        lower = _number(card.get("Крепость от, %"))
        upper = _number(card.get("Крепость до, %"))
        if lower is not None and upper is not None:
            alcohol = (lower + upper) / 2
    if alcohol is None:
        return "unknown"
    if alcohol < 10:
        return "low"
    if alcohol < 12:
        return "moderate"
    if alcohol < 14:
        return "standard"
    if alcohol < 16:
        return "high"
    return "very_high"


def _grape_composition(grape: str) -> str:
    """Classify exact Catalog grape text without inventing components."""
    if not grape:
        return "unknown"
    normalized = grape.casefold().replace("ё", "е")
    if "сорта винограда" in normalized:
        return "generic_grapes"
    if re.search(r"[,;/]|\s+и\s+", normalized):
        return "blend"
    return "single_variety"
