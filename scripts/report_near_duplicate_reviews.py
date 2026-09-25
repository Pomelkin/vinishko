#!/usr/bin/env python3
"""Report reviewed candidate and confirmed near-duplicate graph statistics."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "near_duplicates"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def pair(row: dict[str, str]) -> tuple[str, str]:
    return tuple(sorted((row["slug_1"], row["slug_2"])))


def report() -> dict[str, object]:
    registry = read_csv(DATA / "all_candidates.csv")
    reviews = read_csv(DATA / "reviewed_candidates.csv")
    catalog = read_csv(ROOT / "data/strapi/catalog_dataset.csv")
    edges = [pair(row) for row in registry]
    if len(set(edges)) != len(edges):
        raise ValueError("duplicate confirmed pair")
    checked = {pair(row) for row in reviews} | set(edges)
    parent: dict[str, str] = {}

    def root(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in edges:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        parent[root(a)] = root(b)
    components: dict[str, list[str]] = defaultdict(list)
    for slug in parent:
        components[root(slug)].append(slug)
    all_family_pairs = {
        (a, b)
        for slugs in components.values()
        for a, b in itertools.combinations(sorted(slugs), 2)
    }
    component_sizes = Counter(len(slugs) for slugs in components.values())
    verdicts = Counter(row["verdict"] for row in reviews)
    source_confirmed = Counter(row.get("review_source", "legacy") for row in registry)
    image_statuses = Counter(row["Статус изображения"] for row in catalog)
    winery = {row["Slug"]: row["Винодельня"] for row in catalog}
    return {
        "catalog_positions": len(catalog),
        "accepted_reference_positions": sum(status.startswith("ok_") for status in image_statuses.elements()),
        "catalog_image_statuses": dict(sorted(image_statuses.items())),
        "reviewed_candidate_rows": len(reviews),
        "reviewed_candidate_verdicts": dict(sorted(verdicts.items())),
        "confirmed_pairs": len(edges),
        "confirmed_pairs_by_source": dict(sorted(source_confirmed.items())),
        "confirmed_positions": len(parent),
        "confirmed_components": len(components),
        "component_sizes": dict(sorted(component_sizes.items())),
        "largest_component": max(component_sizes, default=0),
        "possible_pairs_within_components": len(all_family_pairs),
        "checked_pairs_within_components": len(all_family_pairs & checked),
        "unreviewed_pairs_within_components": len(all_family_pairs - checked),
        "cross_winery_confirmed_pairs": sum(winery[a] != winery[b] for a, b in edges),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    data = report()
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        for key, value in data.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
