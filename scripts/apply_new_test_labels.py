"""Copy reviewed new_test photos into the archived slug-folder test layout."""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE = DATA / "legacy" / "new_test"
MANIFEST = DATA / "technical" / "new_test_labels.csv"
DESTINATION = DATA / "legacy" / "new_test_labeled"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    with (DATA / "technical/strapi/catalog_dataset.csv").open(encoding="utf-8-sig", newline="") as stream:
        allowed_slugs = {row["Slug"] for row in csv.DictReader(stream)}
    with MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        labels = list(csv.DictReader(stream))

    source_files = {path.name: path for path in SOURCE.glob("*.webp")}
    named = [row["query_file"] for row in labels]
    if len(named) != len(set(named)) or set(named) != set(source_files):
        raise ValueError("The label manifest must contain each source photo exactly once")
    for row in labels:
        label = row["label"]
        if not re.fullmatch(r"[a-z0-9_-]+", label):
            raise ValueError(f"Unsafe label: {label!r}")
        if label not in allowed_slugs and not label.startswith("not_found_"):
            raise ValueError(f"Label is neither a catalog slug nor not_found: {label}")

    DESTINATION.mkdir(exist_ok=True)
    expected_paths = {DESTINATION / row["label"] / row["query_file"] for row in labels}
    existing_paths = {path for path in DESTINATION.rglob("*") if path.is_file()}
    if extra := existing_paths - expected_paths:
        raise ValueError(f"Unexpected preexisting labeled files: {sorted(extra)[:5]}")
    for row in labels:
        source = source_files[row["query_file"]]
        target = DESTINATION / row["label"] / source.name
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            if sha256(source) != sha256(target):
                raise ValueError(f"Existing labeled file differs from source: {target}")
        else:
            shutil.copy2(source, target)
    print(f"Labeled {len(labels)} photos in {DESTINATION}")


if __name__ == "__main__":
    main()
