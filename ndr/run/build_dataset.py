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
NOT_FOUND_REGISTRY_PATH = (
    REPO / "data" / "near_duplicates" / "not_found_candidates.csv"
)
CATALOG_PATH = REPO / "data" / "technical" / "strapi" / "catalog_dataset.csv"
TEST_PATH = REPO / "data" / "legacy" / "test"
MEDIA_PATH = REPO / "data" / "technical" / "strapi" / "img"
GALLERY_PATH = REPO / "data" / "near_duplicates" / "C-visually-close"
NOT_FOUND_PREFIX = "not_found_"
NOT_FOUND = "not_found"

REGISTRY_FIELDS = (
    "candidate_number",
    "slug_1",
    "slug_2",
    "winery_1",
    "winery_2",
    "review_verdict",
    "is_confirmed_near_duplicate",
    "visual_relation",
    "diff_percent",
)

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


def read_csv(
    path: Path,
    expected_fields: tuple[str, ...] | None = None,
) -> list[dict[str, str]]:
    """Read a UTF-8 CSV with a real CSV parser."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if expected_fields is not None and tuple((reader.fieldnames or ())[:len(expected_fields)]) != expected_fields:
            raise DatasetError(
                f"unexpected CSV columns in {relative_path(path)}: {reader.fieldnames}",
            )
        return list(reader)


def require_confirmed_rows(
    rows: list[dict[str, str]],
    source: Path,
) -> None:
    """Require every registry row to carry the reviewed confirmation verdict."""
    if not rows:
        raise DatasetError(f"near-duplicate registry is empty: {relative_path(source)}")
    numbers: set[int] = set()
    for row in rows:
        try:
            number = int(row["candidate_number"])
        except (TypeError, ValueError) as error:
            raise DatasetError(
                f"invalid candidate number in {relative_path(source)}: "
                f"{row['candidate_number']!r}",
            ) from error
        if number <= 0 or number in numbers:
            raise DatasetError(
                f"duplicate or non-positive candidate number in "
                f"{relative_path(source)}: {number}",
            )
        numbers.add(number)
        if (
            row["review_verdict"] != "confirmed_near_duplicate"
            or row["is_confirmed_near_duplicate"].lower() != "true"
        ):
            raise DatasetError(
                f"candidate {number} in {relative_path(source)} is not a "
                "confirmed near-duplicate",
            )


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
    registry_rows = read_csv(REGISTRY_PATH, REGISTRY_FIELDS)
    require_confirmed_rows(registry_rows, REGISTRY_PATH)
    not_found_rows = read_csv(NOT_FOUND_REGISTRY_PATH, REGISTRY_FIELDS)
    require_confirmed_rows(not_found_rows, NOT_FOUND_REGISTRY_PATH)

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

    not_found_by_id: dict[str, dict[str, str]] = {}
    for row in not_found_rows:
        not_found_id = row["slug_1"]
        anchor_slug = row["slug_2"]
        if not not_found_id.startswith(NOT_FOUND_PREFIX):
            raise DatasetError(
                f"not-found candidate slug_1 must start with {NOT_FOUND_PREFIX!r}: "
                f"{not_found_id!r}",
            )
        if not_found_id in catalog_by_slug:
            raise DatasetError(f"not-found ID unexpectedly exists in catalog: {not_found_id}")
        if anchor_slug not in catalog_by_slug:
            raise DatasetError(
                f"not-found candidate {not_found_id} points to a non-catalog slug: "
                f"{anchor_slug}",
            )
        if not adjacency[anchor_slug]:
            raise DatasetError(
                f"not-found candidate {not_found_id} points outside the confirmed "
                f"catalog near-duplicate graph: {anchor_slug}",
            )
        if not_found_id in not_found_by_id:
            raise DatasetError(f"duplicate not-found mapping: {not_found_id}")
        if not (TEST_PATH / not_found_id).is_dir():
            raise DatasetError(f"mapped not-found test directory is missing: {not_found_id}")
        not_found_by_id[not_found_id] = row

    source_queries: list[dict[str, Any]] = []
    for path in sorted(item for item in TEST_PATH.rglob("*") if item.is_file()):
        source_id = path.relative_to(TEST_PATH).parts[0]
        if source_id.startswith(NOT_FOUND_PREFIX):
            mapping = not_found_by_id.get(source_id)
            source_queries.append(
                {
                    "path": path,
                    "source_id": source_id,
                    "expected_slug": NOT_FOUND,
                    "anchor_slug": mapping["slug_2"] if mapping is not None else None,
                    "is_not_found": True,
                },
            )
            continue
        if source_id not in catalog_by_slug:
            raise DatasetError(f"test directory is not a catalog slug: {source_id}")
        source_queries.append(
            {
                "path": path,
                "source_id": source_id,
                "expected_slug": source_id,
                "anchor_slug": source_id,
                "is_not_found": False,
            },
        )
    if not source_queries:
        raise DatasetError("data/legacy/test contains no query files")

    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for query in source_queries:
        if query["is_not_found"]:
            if query["anchor_slug"] is None:
                query["exclusion_reason"] = "no_confirmed_near_duplicate_mapping"
                excluded.append(query)
            else:
                included.append(query)
        elif adjacency[query["anchor_slug"]]:
            included.append(query)
        else:
            query["exclusion_reason"] = "no_confirmed_near_duplicates"
            excluded.append(query)

    included_roots = sorted({query["anchor_slug"] for query in included})
    components = graph_components(adjacency, included_roots)
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
    for number, query in enumerate(included, start=1):
        path = query["path"]
        component = groups_by_slug[query["anchor_slug"]]
        record = {
            "query_id": f"q-{number:06d}",
            "image_path": relative_path(path),
            "image_sha256": file_sha256(path),
            "image_bytes": path.stat().st_size,
            "expected_slug": query["expected_slug"],
            "group_id": group_id(component),
        }
        if query["is_not_found"]:
            record.update(
                {
                    "not_found_id": query["source_id"],
                    "near_duplicate_anchor_slug": query["anchor_slug"],
                },
            )
        manifest_records.append(record)

    excluded_records: list[dict[str, Any]] = []
    for query in excluded:
        path = query["path"]
        record = {
            "image_path": relative_path(path),
            "image_sha256": file_sha256(path),
            "expected_slug": query["expected_slug"],
            "reason": query["exclusion_reason"],
        }
        if query["is_not_found"]:
            record["not_found_id"] = query["source_id"]
        excluded_records.append(record)

    source_not_found = [query for query in source_queries if query["is_not_found"]]
    included_not_found = [query for query in included if query["is_not_found"]]
    excluded_singletons = [query for query in excluded if not query["is_not_found"]]
    excluded_not_found = [query for query in excluded if query["is_not_found"]]

    metadata = {
        "schema_version": 2,
        "selection_rule": (
            "catalog queries whose slug belongs to a connected component in "
            "all_candidates.csv, plus not_found queries mapped by "
            "not_found_candidates.csv to an anchor in such a component"
        ),
        "group_rule": "connected_components_of_confirmed_catalog_registry_edges",
        "not_found_value": NOT_FOUND,
        "source_files": {
            relative_path(REGISTRY_PATH): file_sha256(REGISTRY_PATH),
            relative_path(NOT_FOUND_REGISTRY_PATH): file_sha256(
                NOT_FOUND_REGISTRY_PATH,
            ),
            relative_path(CATALOG_PATH): file_sha256(CATALOG_PATH),
        },
        "counts": {
            "source_query_files": len(source_queries),
            "source_query_slugs": len({query["source_id"] for query in source_queries}),
            "source_not_found_files": len(source_not_found),
            "source_not_found_products": len(
                {query["source_id"] for query in source_not_found}
            ),
            "included_queries": len(manifest_records),
            "included_query_slugs": len({query["source_id"] for query in included}),
            "included_not_found_queries": len(included_not_found),
            "included_not_found_products": len(
                {query["source_id"] for query in included_not_found}
            ),
            "excluded_queries": len(excluded_records),
            "excluded_query_slugs": len({query["source_id"] for query in excluded}),
            "excluded_singleton_queries": len(excluded_singletons),
            "excluded_singleton_slugs": len(
                {query["source_id"] for query in excluded_singletons}
            ),
            "excluded_unmapped_not_found_queries": len(excluded_not_found),
            "excluded_unmapped_not_found_products": len(
                {query["source_id"] for query in excluded_not_found}
            ),
            "mapped_not_found_products": len(not_found_by_id),
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
        f"excluded {counts['excluded_queries']} queries "
        f"({counts['excluded_singleton_queries']} catalog singletons, "
        f"{counts['excluded_unmapped_not_found_queries']} unmapped not_found).",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
