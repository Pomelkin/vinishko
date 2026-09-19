#!/usr/bin/env python3
"""Validate the curated near-duplicate CSV and C-gallery invariants."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

EXPECTED_ROWS = 378
EXPECTED_CATALOG_SLUGS = 2103
EXCLUDED_NUMBERS = {1, 8, 72, 82, 239, 251, 315, 347, 387}
FOLDER_NUMBER = re.compile(r"^(\d{3})_")
INFO_PAIR = re.compile(r"^# (.+)\\\|(.+)$")
PRODUCT_PNG = re.compile(r"^\d+_.+\.png$")


def find_repo(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "AGENTS.md").is_file() and (
            path / "data/strapi/catalog_dataset.csv"
        ).is_file():
            return path
    raise RuntimeError("repository root not found")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate(repo: Path) -> list[str]:
    errors: list[str] = []
    catalog_path = repo / "data/strapi/catalog_dataset.csv"
    registry_path = repo / "data/near_duplicates/all_candidates.csv"
    gallery_path = repo / "data/near_duplicates/C-visually-close"

    if not registry_path.is_file():
        return [f"missing registry: {registry_path}"]
    if not gallery_path.is_dir():
        return [f"missing gallery: {gallery_path}"]

    catalog_rows = read_csv(catalog_path)
    catalog_slugs = {row.get("Slug", "").strip() for row in catalog_rows}
    catalog_slugs.discard("")
    if len(catalog_slugs) != EXPECTED_CATALOG_SLUGS:
        errors.append(
            f"catalog has {len(catalog_slugs)} unique slugs; expected {EXPECTED_CATALOG_SLUGS}"
        )

    rows = read_csv(registry_path)
    if len(rows) != EXPECTED_ROWS:
        errors.append(f"registry has {len(rows)} rows; expected {EXPECTED_ROWS}")

    rows_by_number: dict[int, dict[str, str]] = {}
    unordered_pairs: set[frozenset[str]] = set()
    for line_number, row in enumerate(rows, start=2):
        try:
            number = int(row.get("candidate_number", ""))
        except ValueError:
            errors.append(f"CSV line {line_number}: invalid candidate_number")
            continue
        if number in rows_by_number:
            errors.append(f"duplicate candidate_number: {number}")
        rows_by_number[number] = row
        if number in EXCLUDED_NUMBERS:
            errors.append(f"reviewed exclusion present in CSV: {number}")

        slug_1 = row.get("slug_1", "")
        slug_2 = row.get("slug_2", "")
        for slug in (slug_1, slug_2):
            if slug not in catalog_slugs:
                errors.append(f"candidate {number}: slug absent from catalog_dataset.csv: {slug}")
        pair = frozenset((slug_1, slug_2))
        if len(pair) != 2:
            errors.append(f"candidate {number}: pair does not contain two distinct slugs")
        elif pair in unordered_pairs:
            errors.append(f"duplicate unordered pair at candidate {number}")
        unordered_pairs.add(pair)

        if row.get("review_verdict") != "confirmed_near_duplicate":
            errors.append(f"candidate {number}: non-confirmed review_verdict")
        if row.get("is_confirmed_near_duplicate", "").lower() != "true":
            errors.append(f"candidate {number}: confirmation flag is not true")

    folders_by_number: dict[int, Path] = {}
    for folder in sorted(path for path in gallery_path.iterdir() if path.is_dir()):
        match = FOLDER_NUMBER.match(folder.name)
        if not match:
            errors.append(f"gallery folder has no three-digit prefix: {folder.name}")
            continue
        number = int(match.group(1))
        if number in folders_by_number:
            errors.append(f"duplicate gallery folder number: {number}")
        folders_by_number[number] = folder

    if len(folders_by_number) != EXPECTED_ROWS:
        errors.append(f"gallery has {len(folders_by_number)} folders; expected {EXPECTED_ROWS}")

    missing_folders = sorted(rows_by_number.keys() - folders_by_number.keys())
    extra_folders = sorted(folders_by_number.keys() - rows_by_number.keys())
    if missing_folders:
        errors.append(f"CSV candidates without folders: {missing_folders}")
    if extra_folders:
        errors.append(f"folders without CSV candidates: {extra_folders}")

    for number in sorted(rows_by_number.keys() & folders_by_number.keys()):
        row = rows_by_number[number]
        folder = folders_by_number[number]
        info_path = folder / "info.md"
        if not info_path.is_file():
            errors.append(f"candidate {number}: missing info.md")
            continue
        first_line = info_path.read_text(encoding="utf-8").splitlines()[0]
        match = INFO_PAIR.match(first_line)
        if not match:
            errors.append(f"candidate {number}: malformed info.md header")
        elif match.groups() != (row.get("slug_1", ""), row.get("slug_2", "")):
            errors.append(f"candidate {number}: CSV/info.md pair mismatch")

        product_pngs = [path for path in folder.iterdir() if PRODUCT_PNG.match(path.name)]
        if len(product_pngs) != 2:
            errors.append(f"candidate {number}: found {len(product_pngs)} bottle PNGs, expected 2")
        for required in ("_compare.png", "_zoom.png"):
            if not (folder / required).is_file():
                errors.append(f"candidate {number}: missing {required}")

    if (repo / "data/near_duplicates/index.md").exists():
        errors.append("Markdown aggregate index.md must not exist")
    if (repo / "data/near_duplicates_full_build").exists():
        errors.append("temporary near_duplicates_full_build was not removed")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, help="repository root; auto-detected by default")
    args = parser.parse_args()
    repo = args.repo.resolve() if args.repo else find_repo(Path(__file__).resolve())
    errors = validate(repo)
    if errors:
        print(f"FAIL: {len(errors)} near-duplicate invariant error(s)", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(
        f"OK: {EXPECTED_ROWS} confirmed pairs; {EXPECTED_ROWS} folders; "
        f"{EXPECTED_CATALOG_SLUGS} catalog slugs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
