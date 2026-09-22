"""Offline checks for the golden contract; factual/style review remains manual.

Run from the repository root: python ai_somelier/golden/validate.py
An optional positional argument selects another dataset file.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
PROMPT_CARD_FIELDS = (
    "Название вина", "Категория", "Цвет", "Регион", "Сорт винограда",
    "Описание", "Винодельня", "Slug", "Винтаж", "Крепость, %",
    "Крепость от, %", "Крепость до, %", "Сахар", "Игристое",
    "Метод игристого", "Креплёное", "Безалкогольное", "Органическое",
    "Выдержка или резерв",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def keys(value: object, expected: set[str], location: str) -> None:
    require(isinstance(value, dict), f"{location}: expected object")
    require(set(value) == expected, f"{location}: unexpected or missing keys")


def nonempty(value: object, location: str) -> None:
    require(isinstance(value, str) and bool(value.strip()), f"{location}: empty/non-string text")
    require(value == value.strip(), f"{location}: surrounding whitespace")
    require(value not in {"X", "Y"}, f"{location}: unfilled placeholder")


def validate(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    keys(data, {"schema_version", "dataset_id", "language", "catalog", "cases"}, "root")
    require(data["schema_version"] == "1.0", "Unsupported schema version")
    require(data["dataset_id"] == "ai_somelier_golden_v1", "Unexpected dataset ID")
    require(data["language"] == "ru", "Unexpected language")
    meta = data["catalog"]
    keys(meta, {"path", "sha256", "card_fields"}, "catalog")
    require(meta["path"] == "data/strapi/catalog_dataset.csv", "Unexpected Catalog path")
    require(meta["card_fields"] == list(PROMPT_CARD_FIELDS), "Catalog field contract differs")
    catalog_path = ROOT / meta["path"]
    require(hashlib.sha256(catalog_path.read_bytes()).hexdigest() == meta["sha256"], "Catalog SHA-256 differs")
    with catalog_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    catalog = {row["Slug"]: row for row in rows}
    require(len(catalog) == len(rows) and "" not in catalog, "Duplicate/empty Catalog Slug")

    cases = data["cases"]
    require(isinstance(cases, list) and len(cases) == 10, "Expected 10 cases")
    ids = set()
    lengths = Counter()
    first_lengths = []
    for case in cases:
        keys(case, {"id", "title", "coverage", "input", "golden"}, "case")
        cid = case["id"]
        nonempty(cid, "case.id")
        require(cid not in ids, f"Duplicate case ID: {cid}")
        ids.add(cid)
        nonempty(case["title"], f"{cid}.title")
        keys(case["coverage"], {"follow_up_turns", "intents"}, f"{cid}.coverage")
        intents = case["coverage"]["intents"]
        require(isinstance(intents, list) and bool(intents), f"{cid}: expected intents")
        for intent in intents:
            nonempty(intent, f"{cid}.intent")
        require(len(set(intents)) == len(intents), f"{cid}: duplicate intent")
        keys(case["input"], {"wine", "candidates"}, f"{cid}.input")
        candidates = case["input"]["candidates"]
        require(isinstance(candidates, list) and len(candidates) == 5, f"{cid}: expected 5 candidates")
        slugs = set()
        for card in [case["input"]["wine"], *candidates]:
            keys(card, set(PROMPT_CARD_FIELDS), f"{cid}.card")
            require(all(isinstance(v, str) for v in card.values()), f"{cid}: non-string card value")
            slug = card["Slug"]
            require(slug in catalog and slug not in slugs, f"{cid}: unknown/repeated Slug {slug}")
            slugs.add(slug)
            for field in PROMPT_CARD_FIELDS:
                require(card[field] == catalog[slug][field], f"{cid}/{slug}: Catalog mismatch in {field}")

        golden = case["golden"]
        keys(golden, {"first_turn", "turns"}, f"{cid}.golden")
        first = golden["first_turn"]
        keys(first, {"content", "suggestions"}, f"{cid}.first_turn")
        content = first["content"]
        nonempty(content, f"{cid}.content")
        first_lengths.append(len(content))
        require(350 <= len(content) <= 750, f"{cid}: first turn has {len(content)} characters")
        require(len(content.split()) <= 100, f"{cid}: first turn exceeds 100 words")
        require(content.count("?") == 1 and content.endswith("Чем я могу вам помочь?"), f"{cid}: closing question differs")
        # Suitable for this corpus: decimal commas, no dotted abbreviations in openings.
        sentences = re.findall(r"[.!?](?=\s|$)", content)
        require(4 <= len(sentences) <= 5, f"{cid}: expected 4–5 sentences, found {len(sentences)}")
        suggestions = first["suggestions"]
        require(isinstance(suggestions, list) and len(suggestions) == 2, f"{cid}: expected 2 suggestions")
        for suggestion in suggestions:
            nonempty(suggestion, f"{cid}.suggestion")
            require(len(suggestion) < 30, f"{cid}: suggestion too long")
        require(suggestions[0] != suggestions[1], f"{cid}: repeated suggestion")
        turns = golden["turns"]
        require(isinstance(turns, list), f"{cid}: turns must be an array")
        count = case["coverage"]["follow_up_turns"]
        require(type(count) is int and count == len(turns), f"{cid}: follow-up count differs")
        lengths[count] += 1
        for index, turn in enumerate(turns, 1):
            keys(turn, {"user", "assistant"}, f"{cid}.turn{index}")
            for role, content in turn.items():
                nonempty(content, f"{cid}.turn{index}.{role}")

    require(lengths == {0: 1, 1: 5, 2: 3, 3: 1}, "Dialogue length distribution differs")
    print("OK: 10 cases, 14 follow-ups, 60 Catalog cards; SHA-256 and structure match.")
    print(f"First turns: {min(first_lengths)}–{max(first_lengths)} characters; sentence, word and suggestion limits pass.")
    print("Factual accuracy, natural wording and scenario coverage require manual review.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, default=ROOT / "ai_somelier/golden_dataset.json")
    args = parser.parse_args()
    try:
        validate(args.dataset)
    except (ValueError, OSError) as error:
        parser.exit(1, f"ERROR: {error}\n")
