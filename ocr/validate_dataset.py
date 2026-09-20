"""Validate OCR dataset coverage, hashes, Catalog links, and feature schema."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = Path(__file__).resolve().parent / "dataset.jsonl"
CATALOG_PATH = ROOT / "data" / "strapi" / "catalog_dataset.csv"
TEST_DIR = ROOT / "data" / "test"
ALLOWED_ROLES = {"identity", "disambiguation", "secondary"}
ALLOWED_KINDS = {
    "abv",
    "appellation",
    "brand",
    "color",
    "grape",
    "line",
    "method",
    "producer",
    "product",
    "region",
    "sugar",
    "vintage",
    "wine_type",
}


class ValidationError(RuntimeError):
    """Raised when a persisted OCR annotation violates an invariant."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL while preserving line-number diagnostics."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValidationError(f"Line {line_number} must contain a JSON object")
            records.append(record)
    return records


def catalog_slugs() -> set[str]:
    """Read the authoritative set of Catalog slugs."""
    with CATALOG_PATH.open(encoding="utf-8-sig", newline="") as source:
        return {row["Slug"] for row in csv.DictReader(source) if row.get("Slug")}


def validate_feature(feature: Any, query_id: str) -> None:
    """Validate a gold feature and all optional aliases."""
    if not isinstance(feature, dict):
        raise ValidationError(f"{query_id}: each gold feature must be an object")
    if feature.get("kind") not in ALLOWED_KINDS:
        raise ValidationError(
            f"{query_id}: invalid feature kind {feature.get('kind')!r}"
        )
    if feature.get("role") not in ALLOWED_ROLES:
        raise ValidationError(
            f"{query_id}: invalid feature role {feature.get('role')!r}"
        )
    if not isinstance(feature.get("value"), str) or not feature["value"].strip():
        raise ValidationError(f"{query_id}: feature value must be a non-empty string")
    aliases = feature.get("aliases", [])
    if not isinstance(aliases, list) or not all(
        isinstance(alias, str) and alias.strip() for alias in aliases
    ):
        raise ValidationError(f"{query_id}: aliases must be non-empty strings")
    if (
        len({feature["value"].casefold(), *(alias.casefold() for alias in aliases)})
        != len(aliases) + 1
    ):
        raise ValidationError(
            f"{query_id}: duplicate canonical value/alias in {feature['value']!r}"
        )


def validate_image(record: dict[str, Any], query_id: str, image_path: Path) -> None:
    """Validate persisted image metadata against the file signature and bytes."""
    if not image_path.is_file():
        raise ValidationError(
            f"{query_id}: image does not exist: {record['image_path']}"
        )
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if record.get("image_sha256") != digest:
        raise ValidationError(f"{query_id}: SHA-256 mismatch")
    with Image.open(image_path) as image:
        if (
            record.get("image_width") != image.width
            or record.get("image_height") != image.height
        ):
            raise ValidationError(f"{query_id}: image dimensions mismatch")
        if record.get("image_format") != image.format:
            raise ValidationError(f"{query_id}: image format mismatch")


def validate_catalog_link(
    record: dict[str, Any],
    query_id: str,
    folder: str,
    allowed_slugs: set[str],
) -> None:
    """Validate found/not-found status and the exact Catalog slug link."""
    expected_not_found = folder.startswith("not_found_")
    if expected_not_found:
        if (
            record.get("catalog_status") != "not_found"
            or record.get("expected_slug") is not None
        ):
            raise ValidationError(
                f"{query_id}: not_found record has a Catalog slug/status"
            )
        if record.get("not_found_id") != folder:
            raise ValidationError(f"{query_id}: invalid not_found_id")
    else:
        if (
            record.get("catalog_status") != "found"
            or record.get("expected_slug") != folder
        ):
            raise ValidationError(
                f"{query_id}: found record does not match its slug folder"
            )
        if folder not in allowed_slugs:
            raise ValidationError(
                f"{query_id}: slug is absent from the Catalog: {folder}"
            )
        if record.get("not_found_id") is not None:
            raise ValidationError(
                f"{query_id}: found record must have null not_found_id"
            )


def validate_record(
    record: dict[str, Any],
    expected_query_id: str,
    allowed_slugs: set[str],
) -> None:
    """Validate one record against files on disk and the Catalog."""
    query_id = record.get("query_id")
    if query_id != expected_query_id:
        raise ValidationError(
            f"Expected query_id {expected_query_id}, got {query_id!r}"
        )
    image_path_value = record.get("image_path")
    if not isinstance(image_path_value, str):
        raise ValidationError(f"{query_id}: image_path must be a string")
    image_path = ROOT / image_path_value
    try:
        image_path.resolve().relative_to(TEST_DIR.resolve())
    except ValueError as exc:
        raise ValidationError(
            f"{query_id}: path escapes data/test: {image_path_value}"
        ) from exc

    validate_image(record, query_id, image_path)
    validate_catalog_link(record, query_id, image_path.parent.name, allowed_slugs)

    features = record.get("gold_features")
    if not isinstance(features, list) or not features:
        raise ValidationError(f"{query_id}: gold_features must be a non-empty array")
    for feature in features:
        validate_feature(feature, query_id)


def main() -> None:
    """Validate the complete one-record-per-image dataset."""
    records = load_jsonl(DATASET_PATH)
    paths: list[str] = []
    for record in records:
        image_path = record.get("image_path")
        if not isinstance(image_path, str):
            raise ValidationError("Every dataset row must have a string image_path")
        paths.append(image_path)
    if paths != sorted(paths):
        raise ValidationError("Dataset rows must be sorted by image_path")
    if len(paths) != len(set(paths)):
        raise ValidationError("Dataset contains duplicate image_path values")

    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in TEST_DIR.rglob("*")
        if path.is_file()
    }
    annotated_paths = {path for path in paths if isinstance(path, str)}
    if actual_paths != annotated_paths:
        missing = sorted(actual_paths - annotated_paths)
        extra = sorted(annotated_paths - actual_paths)
        raise ValidationError(
            f"Dataset coverage differs from data/test; missing={missing}, extra={extra}"
        )

    allowed_slugs = catalog_slugs()
    for index, record in enumerate(records, start=1):
        validate_record(record, f"ocr-{index:06d}", allowed_slugs)

    not_found_count = sum(record["catalog_status"] == "not_found" for record in records)
    feature_count = sum(len(record["gold_features"]) for record in records)
    print(
        f"OK: {len(records)} images, {feature_count} visible gold features, "
        f"{not_found_count} not_found queries"
    )


if __name__ == "__main__":
    main()
