#!/usr/bin/env python3
"""Export unreviewed full-dump candidates without promoting them to confirmed pairs.

Run after ``semantic_near_duplicates.py --output``. The confirmed registry is read
with a CSV parser and remains the only source of confirmed near-duplicates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
DEFAULT_AUDIT = REPO / "data/near_duplicates/full_dump_semantic_audit.json"
DEFAULT_CONFIRMED = REPO / "data/near_duplicates/all_candidates.csv"
DEFAULT_CATALOG = REPO / "data/technical/strapi/catalog_dataset.csv"
DEFAULT_OUTPUT = REPO / "data/near_duplicates/review_queue.csv"

FIELDS = (
    "review_id", "slug_1", "slug_2", "winery", "image_1", "image_2",
    "edge_similarity", "name_similarity", "candidate_source",
    "semantic_difference", "algorithm_relation", "review_status",
)


def export(audit_path: Path, confirmed_path: Path, catalog_path: Path, output: Path) -> Counter:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    with confirmed_path.open(newline="", encoding="utf-8") as stream:
        confirmed = {
            frozenset((row["slug_1"], row["slug_2"]))
            for row in csv.DictReader(stream)
        }
    with catalog_path.open(newline="", encoding="utf-8") as stream:
        catalog = {row["Slug"]: row for row in csv.DictReader(stream)}

    rows = []
    seen = set()
    for candidate in audit["candidates"]:
        if candidate["relation"] not in {"different_wines", "needs_review"}:
            continue
        slugs = tuple(candidate["slugs"])
        pair = frozenset(slugs)
        if pair in confirmed:
            continue
        if len(pair) != 2 or pair in seen:
            raise ValueError(f"Duplicate or invalid candidate pair: {slugs}")
        seen.add(pair)
        for slug in slugs:
            if slug not in catalog or not catalog[slug]["Статус изображения"].startswith("ok_"):
                raise ValueError(f"Candidate has no accepted Catalog reference: {slug}")
        edge = candidate["edge_similarity"] >= audit["summary"]["edge_threshold"]
        text = candidate["name_similarity"] >= 0.82
        pair_id = hashlib.sha256("\0".join(sorted(slugs)).encode("utf-8")).hexdigest()[:12]
        rows.append({
            "review_id": pair_id,
            "slug_1": slugs[0],
            "slug_2": slugs[1],
            "winery": candidate["winery"],
            "image_1": candidate["files"][0],
            "image_2": candidate["files"][1],
            "edge_similarity": candidate["edge_similarity"],
            "name_similarity": candidate["name_similarity"],
            "candidate_source": "edge+text" if edge and text else "edge" if edge else "text",
            "semantic_difference": " | ".join(candidate["semantic_difference"]),
            "algorithm_relation": candidate["relation"],
            "review_status": "pending",
        })

    rows.sort(key=lambda row: (-float(row["edge_similarity"]), -float(row["name_similarity"]),
                               row["slug_1"], row["slug_2"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return Counter(row["candidate_source"] for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--confirmed", type=Path, default=DEFAULT_CONFIRMED)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    counts = export(args.audit, args.confirmed, args.catalog, args.output)
    print(f"Pending pairs: {sum(counts.values())}; by source: {dict(counts)}")
    print(args.output)


if __name__ == "__main__":
    main()
