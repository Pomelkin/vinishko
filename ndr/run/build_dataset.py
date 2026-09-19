"""Build the prompt-evaluation dataset from authoritative repository sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_OUTPUT = REPO / "ndr" / "dataset"
REGISTRY_PATH = REPO / "data" / "near_duplicates" / "all_candidates.csv"
CATALOG_PATH = REPO / "data" / "strapi" / "catalog_dataset.csv"
TEST_PATH = REPO / "data" / "test"
MEDIA_PATH = REPO / "data" / "strapi" / "img"
GALLERY_PATH = REPO / "data" / "near_duplicates" / "C-visually-close"

CATALOG_FIELDS = {
    "name": "Название вина",
    "category": "Категория",
    "color_description": "Цвет",
    "region": "Регион",
    "grapes": "Сорт винограда",
    "description": "Описание",
    "winery": "Винодельня",
    "original_image_name": "Название фото исходное",
    "image_status": "Статус изображения",
    "vintage": "Винтаж",
    "abv": "Крепость, %",
    "abv_from": "Крепость от, %",
    "abv_to": "Крепость до, %",
    "sugar": "Сахар",
    "sparkling": "Игристое",
    "sparkling_method": "Метод игристого",
    "fortified": "Креплёное",
    "alcohol_free": "Безалкогольное",
    "organic": "Органическое",
    "aging_or_reserve": "Выдержка или резерв",
}
CATALOG_IMAGE_FIELD = "Название фото"


class DatasetError(RuntimeError):
    """Raised when a source invariant needed by the dataset is broken."""


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    """Return a portable repository-relative path."""
    return path.resolve().relative_to(REPO).as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV with a real CSV parser."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def graph_components(
    adjacency: dict[str, set[str]],
    roots: Iterable[str],
) -> list[tuple[str, ...]]:
    """Return unique connected components touched by roots."""
    components: set[tuple[str, ...]] = set()
    for root in roots:
        seen = {root}
        pending = [root]
        while pending:
            current = pending.pop()
            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    pending.append(neighbour)
        components.add(tuple(sorted(seen)))
    return sorted(components)


def group_id(slugs: Iterable[str]) -> str:
    """Build a stable group identifier from its exact slug set."""
    payload = "\n".join(slugs).encode()
    return f"ndg-{hashlib.sha256(payload).hexdigest()[:12]}"


def gallery_reference(
    slug: str,
    registry_rows: list[dict[str, str]],
) -> Path | None:
    """Find a verified-gallery image without deriving registry membership from folders."""
    for row in sorted(registry_rows, key=lambda item: int(item["candidate_number"])):
        if slug not in {row["slug_1"], row["slug_2"]}:
            continue
        number = int(row["candidate_number"])
        folders = sorted(GALLERY_PATH.glob(f"{number:03d}_*"))
        if len(folders) != 1:
            raise DatasetError(
                f"candidate {number} must have exactly one verified gallery folder",
            )
        side = "1" if row["slug_1"] == slug else "2"
        image = folders[0] / f"{side}_{slug}.png"
        if image.is_file():
            return image
    return None


def catalog_record(
    slug: str,
    row: dict[str, str],
    registry_rows: list[dict[str, str]],
) -> dict[str, Any]:
    """Translate one source catalog row into the runner's stable schema."""
    image_name = row[CATALOG_IMAGE_FIELD]
    image_path = MEDIA_PATH / image_name if image_name else None
    image_source = "catalog"
    if image_path is None or not image_path.is_file():
        image_path = gallery_reference(slug, registry_rows)
        image_source = "verified_near_duplicate_gallery"
    if image_path is None or not image_path.is_file():
        raise DatasetError(f"no reference image found for candidate {slug}")

    result: dict[str, Any] = {"slug": slug}
    result.update({target: row[source] for target, source in CATALOG_FIELDS.items()})
    result.update(
        {
            "reference_image_path": relative_path(image_path),
            "reference_image_source": image_source,
            "reference_image_sha256": file_sha256(image_path),
        },
    )
    return result


def json_line(record: dict[str, Any]) -> str:
    """Serialize one deterministic JSONL record."""
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def rendered_dataset() -> tuple[dict[str, str], dict[str, Any]]:
    """Build all output documents in memory and return documents plus statistics."""
    registry_rows = read_csv(REGISTRY_PATH)
    if not registry_rows:
        raise DatasetError("near-duplicate registry is empty")
    for row in registry_rows:
        if (
            row["review_verdict"] != "confirmed_near_duplicate"
            or row["is_confirmed_near_duplicate"].lower() != "true"
        ):
            raise DatasetError(
                f"candidate {row['candidate_number']} is not a confirmed near-duplicate",
            )

    catalog_rows = read_csv(CATALOG_PATH)
    catalog_by_slug: dict[str, dict[str, str]] = {}
    for row in catalog_rows:
        slug = row["Slug"]
        if not slug or slug in catalog_by_slug:
            raise DatasetError(f"empty or duplicate catalog slug: {slug!r}")
        catalog_by_slug[slug] = row

    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in registry_rows:
        left = row["slug_1"]
        right = row["slug_2"]
        if left not in catalog_by_slug or right not in catalog_by_slug:
            raise DatasetError(
                f"candidate {row['candidate_number']} contains a non-catalog slug",
            )
        adjacency[left].add(right)
        adjacency[right].add(left)

    source_queries: list[tuple[Path, str]] = []
    for path in sorted(item for item in TEST_PATH.rglob("*") if item.is_file()):
        slug = path.relative_to(TEST_PATH).parts[0]
        if slug not in catalog_by_slug:
            raise DatasetError(f"test directory is not a catalog slug: {slug}")
        source_queries.append((path, slug))
    if not source_queries:
        raise DatasetError("data/test contains no query files")

    included = [(path, slug) for path, slug in source_queries if adjacency[slug]]
    excluded = [(path, slug) for path, slug in source_queries if not adjacency[slug]]
    included_slugs = sorted({slug for _, slug in included})
    components = graph_components(adjacency, included_slugs)
    groups_by_slug = {
        slug: component for component in components for slug in component
    }

    group_records: list[dict[str, Any]] = []
    for component in components:
        component_set = set(component)
        edges = [
            {
                "candidate_number": int(row["candidate_number"]),
                "slug_1": row["slug_1"],
                "slug_2": row["slug_2"],
                "visual_relation": row["visual_relation"],
                "diff_percent": float(row["diff_percent"]),
            }
            for row in registry_rows
            if row["slug_1"] in component_set and row["slug_2"] in component_set
        ]
        edges.sort(key=lambda item: item["candidate_number"])
        group_records.append(
            {
                "group_id": group_id(component),
                "slugs": list(component),
                "edges": edges,
            },
        )
    group_records.sort(key=lambda item: item["group_id"])

    candidate_slugs = sorted({slug for component in components for slug in component})
    catalog_records = [
        catalog_record(slug, catalog_by_slug[slug], registry_rows)
        for slug in candidate_slugs
    ]

    manifest_records: list[dict[str, Any]] = []
    for number, (path, slug) in enumerate(included, start=1):
        component = groups_by_slug[slug]
        manifest_records.append(
            {
                "query_id": f"q-{number:06d}",
                "image_path": relative_path(path),
                "image_sha256": file_sha256(path),
                "image_bytes": path.stat().st_size,
                "expected_slug": slug,
                "group_id": group_id(component),
            },
        )

    excluded_records = [
        {
            "image_path": relative_path(path),
            "image_sha256": file_sha256(path),
            "expected_slug": slug,
            "reason": "no_confirmed_near_duplicates",
        }
        for path, slug in excluded
    ]

    metadata = {
        "schema_version": 1,
        "selection_rule": (
            "all files from data/test whose expected slug belongs to a connected "
            "component of at least two slugs in all_candidates.csv"
        ),
        "group_rule": "connected_components_of_confirmed_registry_edges",
        "not_found_value": "not_found",
        "source_files": {
            relative_path(REGISTRY_PATH): file_sha256(REGISTRY_PATH),
            relative_path(CATALOG_PATH): file_sha256(CATALOG_PATH),
        },
        "counts": {
            "source_query_files": len(source_queries),
            "source_query_slugs": len({slug for _, slug in source_queries}),
            "included_queries": len(manifest_records),
            "included_query_slugs": len(included_slugs),
            "excluded_singleton_queries": len(excluded_records),
            "excluded_singleton_slugs": len({slug for _, slug in excluded}),
            "groups": len(group_records),
            "candidate_products": len(catalog_records),
        },
    }

    documents = {
        "manifest.jsonl": "".join(f"{json_line(row)}\n" for row in manifest_records),
        "groups.jsonl": "".join(f"{json_line(row)}\n" for row in group_records),
        "catalog.jsonl": "".join(f"{json_line(row)}\n" for row in catalog_records),
        "excluded.jsonl": "".join(f"{json_line(row)}\n" for row in excluded_records),
        "metadata.json": f"{json.dumps(metadata, ensure_ascii=False, indent=2)}\n",
    }
    return documents, metadata


def write_documents(output: Path, documents: dict[str, str]) -> None:
    """Write generated documents atomically within the output directory."""
    output.mkdir(parents=True, exist_ok=True)
    expected_names = set(documents)
    for path in output.iterdir():
        if path.is_file() and path.name not in expected_names:
            raise DatasetError(f"unexpected file in generated dataset directory: {path}")
    for name, content in documents.items():
        destination = output / name
        temporary = output / f".{name}.tmp"
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(destination)


def check_documents(output: Path, documents: dict[str, str]) -> None:
    """Fail when checked-in generated documents are missing or stale."""
    mismatches = []
    for name, expected in documents.items():
        path = output / name
        actual = path.read_text(encoding="utf-8") if path.is_file() else None
        if actual != expected:
            mismatches.append(name)
    if mismatches:
        names = ", ".join(mismatches)
        raise DatasetError(f"dataset is missing or stale: {names}; rebuild it")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify generated files without changing them",
    )
    return parser.parse_args()


def main() -> int:
    """Build or check the generated dataset."""
    args = parse_args()
    try:
        documents, metadata = rendered_dataset()
        if args.check:
            check_documents(args.output, documents)
            action = "valid"
        else:
            write_documents(args.output, documents)
            action = "built"
    except DatasetError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    counts = metadata["counts"]
    print(
        f"Dataset {action}: {counts['included_queries']} queries, "
        f"{counts['groups']} groups, {counts['candidate_products']} candidates; "
        f"excluded {counts['excluded_singleton_queries']} singleton queries.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
