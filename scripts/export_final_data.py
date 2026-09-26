"""Export the reviewed Catalog and labeled test photos into two small packages.

Run from the repository root:

    python scripts/export_final_data.py --output-root data/.export_stage
    python scripts/export_final_data.py --check --output-root data/.export_stage

The export is deliberately write-once: an existing output directory is never
overwritten. Source material stays in data/technical and data/legacy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from PIL import Image


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
ACCEPTED_STATUSES = {"ok_exact", "ok_collision_owner", "ok_shared_same_product"}
IMAGE_EXTENSIONS = {"WEBP": ".webp", "JPEG": ".jpg", "PNG": ".png"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None, path
        return reader.fieldnames, list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(current: str, archived: str) -> Path:
    new = DATA / archived
    old = DATA / current
    path = new if new.exists() else old
    if not path.exists():
        raise FileNotFoundError(f"missing source: {new} or {old}")
    return path


def group_slugs(valid_slugs: set[str]) -> dict[str, str]:
    _, pairs = read_csv(DATA / "near_duplicates/all_candidates.csv")
    parent: dict[str, str] = {}

    def find(slug: str) -> str:
        parent.setdefault(slug, slug)
        if parent[slug] != slug:
            parent[slug] = find(parent[slug])
        return parent[slug]

    for pair in pairs:
        if (pair["review_verdict"] != "confirmed_near_duplicate"
                or pair["is_confirmed_near_duplicate"].lower() != "true"):
            raise ValueError(f"unconfirmed pair in registry: {pair['candidate_number']}")
        a, b = pair["slug_1"], pair["slug_2"]
        if a not in valid_slugs or b not in valid_slugs or a == b:
            raise ValueError(f"invalid near-duplicate pair: {a}, {b}")
        parent[find(a)] = find(b)

    components: dict[str, set[str]] = defaultdict(set)
    for slug in parent:
        components[find(slug)].add(slug)
    groups = {slug: min(members) for members in components.values() for slug in members}
    if len(pairs) != 2097 or len(groups) != 1226 or len(components) != 352:
        raise ValueError("near-duplicate registry counts changed; review the export")
    return groups


def actual_image(source_dir: Path, file_name: str, expected_sha: str) -> Path:
    path = source_dir / file_name
    if path.is_file() and sha256(path) == expected_sha:
        return path
    # A repeated space was lost in one name. Another name collides by case on
    # Windows and opens the wrong file; its reviewed image has a numeric suffix.
    # The persisted SHA-256 is the only safe fallback for either case.
    normalized_stem = " ".join(Path(file_name).stem.split()).casefold()
    nearby = [candidate for candidate in source_dir.iterdir()
              if candidate.is_file()
              and " ".join(candidate.stem.split()).casefold().startswith(normalized_stem)
              and sha256(candidate) == expected_sha]
    if nearby:
        return min(nearby, key=lambda candidate: candidate.name)
    matches = sha_index(source_dir).get(expected_sha, [])
    if not matches:
        raise ValueError(f"cannot resolve accepted image: {file_name!r}")
    return min(matches, key=lambda candidate: candidate.name)


@lru_cache(maxsize=1)
def sha_index(source_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in source_dir.iterdir():
        if path.is_file():
            index[sha256(path)].append(path)
    return index


def build_catalog(output: Path) -> set[str]:
    catalog_file = source_path("strapi/catalog_dataset.csv", "technical/strapi/catalog_dataset.csv")
    image_dir = catalog_file.parent / "img"
    fields, source_rows = read_csv(catalog_file)
    slugs = [row["Slug"] for row in source_rows]
    if len(slugs) != 2103 or len(set(slugs)) != len(slugs):
        raise ValueError("catalog slug count or uniqueness changed")
    groups = group_slugs(set(slugs))

    # The first eight columns are the business card. Columns 12 onward are
    # normalized product attributes. The four intervening fields are image
    # source names, an audit status, and a source SHA-256.
    status_field, sha_field, source_image_field = fields[10], fields[11], fields[9]
    output_fields = [*fields[:8], *fields[12:], "image_filename", "near_duplicate_group_slug"]
    output.mkdir()
    (output / "images").mkdir()
    rows: list[dict[str, str]] = []
    image_count = 0
    for source in source_rows:
        slug = source["Slug"]
        row = {field: source[field] for field in output_fields if field in source}
        row["near_duplicate_group_slug"] = groups.get(slug, "")
        row["image_filename"] = ""
        if source[status_field] in ACCEPTED_STATUSES:
            image = actual_image(image_dir, source[source_image_field], source[sha_field])
            if image.suffix.lower() != ".webp":
                raise ValueError(f"accepted image is not WebP: {image}")
            name = f"{slug}.webp"
            destination = output / "images" / name
            shutil.copyfile(image, destination)
            row["image_filename"] = name
            image_count += 1
        rows.append(row)
    write_csv(output / "catalog.csv", output_fields, rows)
    if image_count != 2055:
        raise ValueError(f"unexpected accepted image count: {image_count}")
    return set(slugs)


def build_test(output: Path, catalog_slugs: set[str]) -> None:
    source_dir = source_path("test_merged", "legacy/test_merged")
    files = sorted(path for path in source_dir.rglob("*") if path.is_file())
    if len(files) != 176:
        raise ValueError(f"unexpected test image count: {len(files)}")
    output.mkdir()
    (output / "images").mkdir()
    rows: list[dict[str, str]] = []
    for number, path in enumerate(files, start=1):
        label = path.relative_to(source_dir).parts[0]
        if label not in catalog_slugs and not label.startswith("not_found_"):
            raise ValueError(f"unknown test label: {label}")
        with Image.open(path) as image:
            extension = IMAGE_EXTENSIONS.get(image.format or "")
            image.verify()
        if extension is None:
            raise ValueError(f"unsupported test image format: {path}")
        name = f"test_{number:06d}{extension}"
        destination = output / "images" / name
        shutil.copyfile(path, destination)
        rows.append({"image_filename": name, "slug": label if label in catalog_slugs else ""})
    write_csv(output / "test.csv", ["image_filename", "slug"], rows)


def check(root: Path) -> None:
    catalog_dir, test_dir = root / "catalog", root / "test"
    catalog_fields, catalog = read_csv(catalog_dir / "catalog.csv")
    test_fields, tests = read_csv(test_dir / "test.csv")
    if len(catalog) != 2103 or len(tests) != 176:
        raise ValueError("unexpected row count")
    if test_fields != ["image_filename", "slug"]:
        raise ValueError("test.csv has unexpected columns")
    if "near_duplicate_group_slug" not in catalog_fields or "image_filename" not in catalog_fields:
        raise ValueError("catalog.csv misses delivery columns")
    slugs = {row["Slug"] for row in catalog}
    if len(slugs) != len(catalog):
        raise ValueError("duplicate catalog slug")
    groups = group_slugs(slugs)
    if any(row["near_duplicate_group_slug"] != groups.get(row["Slug"], "") for row in catalog):
        raise ValueError("incorrect near-duplicate component")
    for folder, rows in [(catalog_dir, catalog), (test_dir, tests)]:
        names = [row["image_filename"] for row in rows if row["image_filename"]]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate image filename in {folder}")
        actual = {path.relative_to(folder / "images").as_posix()
                  for path in (folder / "images").rglob("*") if path.is_file()}
        if actual != set(names):
            raise ValueError(f"image/CSV mismatch in {folder}")
    if sum(bool(row["image_filename"]) for row in catalog) != 2055:
        raise ValueError("incorrect accepted catalog image count")
    if any(row["slug"] and row["slug"] not in slugs for row in tests):
        raise ValueError("test contains non-catalog slug")
    if Counter(bool(row["slug"]) for row in tests) != {True: 135, False: 41}:
        raise ValueError("unexpected test label split")
    print("OK: 2103 catalog rows, 2055 images, 352 groups; 176 test rows (135 catalog, 41 absent)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DATA)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    if args.check:
        check(root)
        return
    if (root / "catalog").exists() or (root / "test").exists():
        raise FileExistsError(f"output already exists in {root}")
    root.mkdir(parents=True, exist_ok=True)
    slugs = build_catalog(root / "catalog")
    build_test(root / "test", slugs)
    check(root)


if __name__ == "__main__":
    main()
