#!/usr/bin/env python3
"""Build a clean, feature-enriched catalogue without changing the source CSV.

The pipeline is intentionally conservative about images:

1. prefer an exact ``Название фото`` -> Strapi upload match;
2. use a unique slug-named binary only when the exact channel is absent;
3. group selected files by SHA-256;
4. if one binary is assigned to semantically different products, keep only an
   owner confirmed in ``image_collision_reviews.json``; without a review all
   references in that conflict are excluded.

This is a general collision gate, not a list of filename substitutions.  It
also works on a fresh raw Strapi dump and on the already renamed media folder.

By default the script writes ``data/strapi/catalog_dataset.csv`` and a JSON
audit next to it.  The original ``strapi_output0709.csv`` is opened read-only.

    python scripts/build_catalog_dataset.py
    python scripts/build_catalog_dataset.py --check-only
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO / "data" / "strapi" / "strapi_output0709.csv"
DEFAULT_OUTPUT = REPO / "data" / "strapi" / "catalog_dataset.csv"
DEFAULT_REPORT = REPO / "data" / "strapi" / "catalog_dataset.report.json"
DEFAULT_IMG_DIR = REPO / "data" / "strapi" / "img"
DEFAULT_JOURNAL = REPO / "scripts" / "rename_journal.json"
DEFAULT_REVIEWS = REPO / "scripts" / "image_collision_reviews.json"

REQUIRED_COLUMNS = (
    "Название вина",
    "Категория",
    "Цвет",
    "Регион",
    "Сорт винограда",
    "Описание",
    "Винодельня",
    "Slug",
    "Название фото",
)

ADDED_COLUMNS = (
    "Название фото исходное",
    "Статус изображения",
    "SHA256 изображения",
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

HEX_SUFFIX_RE = re.compile(r"_[0-9a-f]{10}$")
COPY_SUFFIX_RE = re.compile(r" \((\d+)\)$")
SPACE_RE = re.compile(r"\s+")
YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")

_TRANSLIT_BASE = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "j", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "cz", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}
TRANSLIT = {}
for _lower, _latin in _TRANSLIT_BASE.items():
    TRANSLIT[_lower] = _latin
    TRANSLIT[_lower.upper()] = _latin.capitalize() if _latin else ""


@dataclass
class ImageLink:
    """A candidate media link for one catalogue row."""

    filename: str = ""
    status: str = "missing"
    digest: str = ""


def normalize_cell(value: str | None) -> str:
    """Normalize Unicode and replace every whitespace run, including CR/LF."""
    return SPACE_RE.sub(" ", unicodedata.normalize("NFC", value or "")).strip()


def decamelize(text: str) -> str:
    """Port the relevant part of ``@sindresorhus/slugify`` decamelization."""
    text = re.sub(r"([A-Z]{2,})(\d+)", r"\1 \2", text)
    text = re.sub(r"([a-z\d]+)([A-Z]{2,})", r"\1 \2", text)
    text = re.sub(r"([a-z\d])([A-Z])", r"\1 \2", text)
    return re.sub(r"([A-Z]+)([A-Z][a-z\d]+)", r"\1 \2", text)


def name_to_slug(name: str) -> str:
    """Reproduce Strapi's upload basename normalization."""
    transliterated = "".join(TRANSLIT.get(char, char) for char in unicodedata.normalize("NFC", name))
    slug = re.sub(r"[^a-zA-Z\d]+", "_", decamelize(transliterated))
    return re.sub(r"_+", "_", slug).strip("_")


def slug_key(name: str) -> str:
    """Normalize a filename/slug for the conservative slug fallback channel."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def without_extension(filename: str) -> str:
    """Return a basename without only its final extension."""
    return os.path.splitext(filename)[0]


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read and validate the original CSV."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"В {path} нет колонок: {', '.join(missing)}")
        return fieldnames, list(reader)


def normalize_and_deduplicate(
    fieldnames: list[str], rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Clean all cells and retain the first occurrence of each complete row."""
    output: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    changed_cells = 0
    multiline_cells = 0
    for raw in rows:
        cleaned: dict[str, str] = {}
        for field in fieldnames:
            value = raw.get(field, "") or ""
            if "\n" in value or "\r" in value:
                multiline_cells += 1
            normalized = normalize_cell(value)
            changed_cells += normalized != value
            cleaned[field] = normalized
        key = tuple(cleaned[field] for field in fieldnames)
        if key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output, {
        "input_rows": len(rows),
        "output_rows": len(output),
        "removed_complete_duplicates": len(rows) - len(output),
        "normalized_cells": changed_cells,
        "multiline_cells_seen": multiline_cells,
    }


def load_journal(journal_path: Path) -> dict[str, str]:
    """Return current media filename -> original Strapi filename."""
    if not journal_path.exists():
        return {}
    payload = json.loads(journal_path.read_text(encoding="utf-8"))
    return {record["dst"]: record["src"] for record in payload.get("renames", [])}


def candidate_rank(filename: str, requested: str) -> tuple[int, int, str]:
    """Prefer the unsuffixed, exact renamed file, then deterministic copies."""
    current_stem = without_extension(filename)
    requested_stem = without_extension(requested)
    exact = filename.casefold() == requested.casefold()
    copy_match = COPY_SUFFIX_RE.sub("", current_stem).casefold() == requested_stem.casefold()
    copy_number = COPY_SUFFIX_RE.search(current_stem)
    return (
        0 if exact else 1 if copy_match else 2,
        int(copy_number.group(1)) if copy_number else 1,
        filename.casefold(),
    )


def build_media_indices(
    img_dir: Path, journal_path: Path,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Index both raw Strapi names and names produced by the rename script."""
    current_to_source = load_journal(journal_path)
    by_photo_key: dict[str, set[str]] = collections.defaultdict(set)
    by_slug_key: dict[str, set[str]] = collections.defaultdict(set)
    if not img_dir.is_dir():
        return {}, {}

    for path in img_dir.iterdir():
        if not path.is_file() or path.name.startswith("."):
            continue
        current = path.name
        source = current_to_source.get(current, current)
        source_stem = HEX_SUFFIX_RE.sub("", without_extension(source))
        display_stem = COPY_SUFFIX_RE.sub("", without_extension(current))

        # A raw upload has ``nameToSlug(photo name)_<hash>``.  An already
        # renamed file has the human-readable photo name.  Index both forms.
        by_photo_key[source_stem].add(current)
        by_photo_key[name_to_slug(display_stem)].add(current)
        by_slug_key[slug_key(source_stem)].add(current)
        by_slug_key[slug_key(display_stem)].add(current)

    return (
        {key: sorted(values) for key, values in by_photo_key.items()},
        {key: sorted(values) for key, values in by_slug_key.items()},
    )


def link_images(
    rows: list[dict[str, str]], img_dir: Path, journal_path: Path,
) -> dict[str, ImageLink]:
    """Choose at most one deterministic image candidate per slug."""
    by_photo_key, by_slug_name = build_media_indices(img_dir, journal_path)
    links: dict[str, ImageLink] = {}
    digest_cache: dict[str, str] = {}

    for row in rows:
        slug = row["Slug"]
        requested = row["Название фото"]
        photo_key = name_to_slug(without_extension(requested))
        candidates = by_photo_key.get(photo_key, [])
        status = "ok_exact"
        if not candidates:
            candidates = by_slug_name.get(slug_key(slug), [])
            status = "ok_slug_fallback"
            fallback_digests = {
                hashlib.sha256((img_dir / filename).read_bytes()).hexdigest()
                for filename in candidates
            }
            if len(fallback_digests) > 1:
                links[slug] = ImageLink(status="excluded_ambiguous_slug_fallback")
                continue
        if not candidates:
            links[slug] = ImageLink()
            continue
        chosen = min(candidates, key=lambda filename: candidate_rank(filename, requested))
        if chosen not in digest_cache:
            digest_cache[chosen] = hashlib.sha256((img_dir / chosen).read_bytes()).hexdigest()
        links[slug] = ImageLink(chosen, status, digest_cache[chosen])
    return links


def searchable_text(row: dict[str, str]) -> str:
    """Text source explicitly allowed for derived product features."""
    return f"{row['Название вина']} {row['Slug'].replace('-', ' ')}".lower().replace("ё", "е")


def has_pattern(text: str, *patterns: str) -> bool:
    """Return whether any case-folded feature expression occurs."""
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def extract_sugar(row: dict[str, str]) -> str:
    """Extract a sugar/dosage category from name and slug, most specific first."""
    text = searchable_text(row)
    rules = (
        ("брют натюр", (r"\bzero\s+dosage\b", r"\bbrut\s+nature\b", r"\bбрют\s+натюр\b", r"\bдозаж\s+зеро\b")),
        ("экстра брют", (r"\bextra\s+brut\b", r"\bekstra\s+bryut\b", r"\bэкстра\s+брют\b")),
        ("экстра сухое", (r"\bextra\s+dry\b", r"\bekstra\s+suhoe\b", r"\bэкстра\s+сухое\b")),
        ("полусухое", (r"\bsemi\s*dry\b", r"\bpolusuhoe\b", r"\bполусухое\b")),
        ("полусладкое", (r"\bsemi\s*sweet\b", r"\bpolusladkoe\b", r"\bполусладкое\b")),
        ("десертное", (r"\bdessert\b", r"\bdesertnoe\b", r"\bдесертн\w*\b")),
        ("брют", (r"\bbrut\b", r"\bbryut\b", r"\bбрют\b")),
        ("сладкое", (r"\bsweet\b", r"\bsladkoe\b", r"\bсладкое\b")),
        ("сухое", (r"\bdry\b", r"\bsuhoe\b", r"\bсухое\b")),
    )
    return next((label for label, patterns in rules if has_pattern(text, *patterns)), "")


def decode_abv_token(token: str) -> float | None:
    """Decode slug ABV notation: ``13`` -> 13 and ``135`` -> 13.5."""
    number = int(token)
    if 7 <= number <= 25:
        return float(number)
    if 70 <= number <= 250:
        return number / 10
    return None


def extract_abv(slug: str, name: str = "") -> tuple[float | None, float | None]:
    """Extract a terminal ABV or ABV range, excluding years/copy suffixes."""
    match = re.search(r"-(\d{1,3})(?:-(\d{1,3}))?$", slug)
    if not match:
        return None, None
    name_year = YEAR_RE.search(name)
    if not match.group(2) and name_year and name_year.group(1).endswith(match.group(1)):
        # ``daniel-22`` is a short vintage suffix because the name says 2022,
        # not an implausible 22% ABV inferred from the same digits.
        return None, None
    low = decode_abv_token(match.group(1))
    high = decode_abv_token(match.group(2)) if match.group(2) else low
    if low is None or high is None or low > high:
        return None, None
    return low, high


def format_number(value: float | None) -> str:
    """Format a decimal without locale-dependent separators."""
    if value is None:
        return ""
    return f"{value:.1f}".rstrip("0").rstrip(".")


def extract_vintage(row: dict[str, str]) -> str:
    """Prefer an explicit year in the display name, then in the slug."""
    name_match = YEAR_RE.search(row["Название вина"])
    slug_match = YEAR_RE.search(row["Slug"])
    match = name_match or slug_match
    return match.group(1) if match else ""


def sparkling_features(row: dict[str, str], sugar: str) -> tuple[str, str]:
    """Extract explicit sparkling status and production method."""
    text = searchable_text(row)
    is_sparkling = sugar in {"брют натюр", "экстра брют", "брют"} or has_pattern(
        text,
        r"\bигрист\w*\b", r"\bsparkling\b", r"\bshampan\w*\b", r"\bшампан\w*\b",
        r"\bspumante\b", r"\bprosecco\b", r"\bsekt\b", r"\bfrizzante\b",
        r"\bpet\s*nat\b", r"\bpetnat\b", r"\bпет\s*нат\b", r"\bпетнат\b",
    )
    method = ""
    if has_pattern(text, r"\bpet\s*nat\b", r"\bpetnat\b", r"\bпет\s*нат\b", r"\bпетнат\b"):
        method = "петнат"
    elif has_pattern(text, r"\bmethod\s+classic\b", r"\bmethode\s+traditionnelle\b", r"\bклассическ\w*\s+метод\w*\b"):
        method = "классический"
    elif has_pattern(text, r"\bcharmat\b", r"\bшарма\b", r"\bрезервуарн\w*\b"):
        method = "резервуарный"
    return ("да" if is_sparkling else "не определено"), method


def extract_features(row: dict[str, str]) -> dict[str, str]:
    """Build all columns derivable from the catalogue name and slug."""
    text = searchable_text(row)
    sugar = extract_sugar(row)
    abv_low, abv_high = extract_abv(row["Slug"], row["Название вина"])
    sparkling, method = sparkling_features(row, sugar)
    if abv_low is None:
        abv_display = ""
    elif abv_low == abv_high:
        abv_display = format_number(abv_low)
    else:
        abv_display = f"{format_number(abv_low)}-{format_number(abv_high)}"
    return {
        "Винтаж": extract_vintage(row),
        "Крепость, %": abv_display,
        "Крепость от, %": format_number(abv_low),
        "Крепость до, %": format_number(abv_high),
        "Сахар": sugar,
        "Игристое": sparkling,
        "Метод игристого": method,
        "Креплёное": "да" if has_pattern(
            text, r"\bкреплен\w*\b", r"\bкреплён\w*\b", r"\bfortified\b",
            r"\bпортвейн\w*\b", r"\bport\s+wine\b", r"\bмадер\w*\b", r"\bхерес\w*\b",
        ) else "",
        "Безалкогольное": "да" if has_pattern(
            text, r"\bбезалкогольн\w*\b", r"\bnon\s*alcoholic\b", r"\balcohol\s*free\b",
        ) else "",
        "Органическое": "да" if has_pattern(
            text, r"\borganic\b", r"\borganik\b", r"\bорганик\w*\b", r"\bбиодинами\w*\b",
        ) else "",
        "Выдержка или резерв": "да" if has_pattern(
            text, r"\breserv(?:e|a|va)?\b", r"\briserva\b", r"\brezerv\b",
            r"\bвыдерж\w*\b", r"\bbarrel\b", r"\bbarrik\b", r"\bбочк\w*\b",
        ) else "",
    }


def normalized_words(text: str) -> str:
    """Normalize product text for compatibility checks, not fuzzy identity."""
    text = text.lower().replace("ё", "е")
    text = YEAR_RE.sub(" ", text)
    text = re.sub(
        r"\b(вино|wine|белое|красное|розовое|оранжевое|сухое|полусухое|полусладкое|сладкое|"
        r"брют|экстра|extra|brut|dry|semi|suhoe|polusuhoe|polusladkoe|sladkoe)\b",
        " ",
        text,
    )
    return SPACE_RE.sub(" ", re.sub(r"[^\w]+", " ", text)).strip()


def same_product(left: dict[str, str], right: dict[str, str]) -> bool:
    """Decide whether sharing one binary is semantically plausible."""
    exact_fields = ("Винодельня", "Категория", "Сорт винограда")
    if any(normalized_words(left[field]) != normalized_words(right[field]) for field in exact_fields):
        return False
    if normalized_words(left["Название вина"]) != normalized_words(right["Название вина"]):
        return False
    left_year, right_year = extract_vintage(left), extract_vintage(right)
    if left_year and right_year and left_year != right_year:
        return False
    left_sugar, right_sugar = extract_sugar(left), extract_sugar(right)
    return not (left_sugar and right_sugar and left_sugar != right_sugar)


def load_collision_reviews(path: Path) -> dict[str, set[str]]:
    """Load manually label-verified owners, keyed by immutable content hash."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        digest: set(review.get("owner_slugs", []))
        for digest, review in payload.get("reviews", {}).items()
    }


def apply_collision_gate(
    rows: list[dict[str, str]], links: dict[str, ImageLink], reviews_path: Path,
) -> list[dict[str, object]]:
    """Reject incompatible shared binaries and return an audit trail."""
    by_slug = {row["Slug"]: row for row in rows}
    digest_slugs: dict[str, list[str]] = collections.defaultdict(list)
    for slug, link in links.items():
        if link.digest:
            digest_slugs[link.digest].append(slug)
    reviews = load_collision_reviews(reviews_path)
    conflicts: list[dict[str, object]] = []

    for digest, slugs in sorted(digest_slugs.items()):
        if len(slugs) < 2:
            continue
        compatible = all(
            same_product(by_slug[left], by_slug[right])
            for index, left in enumerate(slugs)
            for right in slugs[index + 1:]
        )
        if compatible:
            for slug in slugs:
                links[slug].status = "ok_shared_same_product"
            continue

        reviewed_owners = reviews.get(digest, set())
        owners = sorted(set(slugs) & reviewed_owners)
        excluded = []
        for slug in slugs:
            if slug in owners:
                links[slug].status = "ok_collision_owner"
            else:
                links[slug] = ImageLink(status="excluded_image_collision")
                excluded.append(slug)
        conflicts.append({
            "sha256": digest,
            "candidate_slugs": sorted(slugs),
            "retained_owner_slugs": owners,
            "excluded_slugs": sorted(excluded),
            "reviewed": digest in reviews,
        })
    return conflicts


def enrich_rows(rows: list[dict[str, str]], links: dict[str, ImageLink]) -> list[dict[str, str]]:
    """Attach corrected image fields and derived product attributes."""
    output = []
    for source in rows:
        row = dict(source)
        original_photo = row["Название фото"]
        link = links[row["Slug"]]
        row["Название фото исходное"] = original_photo
        row["Название фото"] = link.filename
        row["Статус изображения"] = link.status
        row["SHA256 изображения"] = link.digest
        row.update(extract_features(row))
        output.append(row)
    return output


def validate_dataset(
    rows: list[dict[str, str]], fieldnames: list[str], img_dir: Path,
) -> dict[str, object]:
    """Fail fast when the generated dataset violates its contract."""
    errors = []
    tuples = [tuple(row.get(field, "") for field in fieldnames) for row in rows]
    if len(tuples) != len(set(tuples)):
        errors.append("в выходном CSV остались полные дубли")
    slugs = [row["Slug"] for row in rows]
    if len(slugs) != len(set(slugs)):
        errors.append("Slug не уникален после удаления полных дублей")
    if any("\n" in value or "\r" in value for row in rows for value in row.values()):
        errors.append("в выходном CSV остались переносы строк в ячейках")
    missing_files = sorted({
        row["Название фото"] for row in rows
        if row["Название фото"] and not (img_dir / row["Название фото"]).is_file()
    })
    if missing_files:
        errors.append(f"ссылки на отсутствующие изображения: {missing_files[:5]}")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "rows_unique": True,
        "slugs_unique": True,
        "cells_without_line_breaks": True,
        "all_nonempty_image_links_exist": True,
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write the derived dataset without touching the input file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    img_dir: Path,
    journal_path: Path,
    reviews_path: Path,
    *,
    check_only: bool = False,
) -> dict[str, object]:
    """Run the complete catalogue build and return its machine-readable audit."""
    original_digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    source_fields, raw_rows = read_rows(input_path)
    clean_rows, cleaning = normalize_and_deduplicate(source_fields, raw_rows)
    links = link_images(clean_rows, img_dir, journal_path)
    conflicts = apply_collision_gate(clean_rows, links, reviews_path)
    output_rows = enrich_rows(clean_rows, links)
    output_fields = source_fields + [field for field in ADDED_COLUMNS if field not in source_fields]
    validation = validate_dataset(output_rows, output_fields, img_dir)
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != original_digest:
        raise RuntimeError("Исходный каталог изменился во время сборки")

    image_statuses = collections.Counter(row["Статус изображения"] for row in output_rows)
    feature_counts = {
        field: sum(bool(row[field]) and row[field] != "не определено" for row in output_rows)
        for field in ADDED_COLUMNS[3:]
    }
    report: dict[str, object] = {
        "rule": {
            "rows": "normalize all whitespace, then remove complete duplicate rows",
            "images": (
                "exact photo-name match; unique slug-named binary only when exact is absent; "
                "SHA-256 collision gate; retain only label-reviewed owner for incompatible products"
            ),
            "features": "derived only from Название вина and Slug; unknown values stay empty/not determined",
        },
        "source": str(input_path),
        "source_sha256": original_digest,
        "output": str(output_path),
        "cleaning": cleaning,
        "image_statuses": dict(sorted(image_statuses.items())),
        "image_conflicts": conflicts,
        "feature_nonempty_counts": feature_counts,
        "validation": validation,
        "check_only": check_only,
    }
    if not check_only:
        write_csv(output_path, output_fields, output_rows)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="исходный CSV (только чтение)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="новый очищенный CSV")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="JSON-аудит сборки")
    parser.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR, help="медиа-дамп Strapi")
    parser.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL, help="журнал переименования медиа")
    parser.add_argument("--collision-reviews", type=Path, default=DEFAULT_REVIEWS, help="проверенные владельцы общих файлов")
    parser.add_argument("--check-only", action="store_true", help="проверить сборку, не записывая файлы")
    args = parser.parse_args()

    try:
        report = build_dataset(
            args.input,
            args.output,
            args.report,
            args.img_dir,
            args.journal,
            args.collision_reviews,
            check_only=args.check_only,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1

    cleaning = report["cleaning"]
    print("Правило изображений:")
    print(f"  {report['rule']['images']}")
    print(
        f"Строк: {cleaning['input_rows']} -> {cleaning['output_rows']} "
        f"(удалено полных дублей: {cleaning['removed_complete_duplicates']})"
    )
    print(f"Статусы изображений: {json.dumps(report['image_statuses'], ensure_ascii=False)}")
    print(f"Конфликтных SHA-групп исправлено: {len(report['image_conflicts'])}")
    if args.check_only:
        print("Проверка пройдена; файлы не записывались.")
    else:
        print(f"Датасет: {args.output}")
        print(f"Аудит: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
