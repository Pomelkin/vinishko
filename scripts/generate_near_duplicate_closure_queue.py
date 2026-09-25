#!/usr/bin/env python3
"""List unreviewed pairs inside already confirmed near-duplicate components."""

from __future__ import annotations

import csv
import argparse
import hashlib
import itertools
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BATCHES = ROOT / "data" / "near_duplicates" / "review_batches"
FIELDS = ["review_id", "slug_1", "slug_2", "winery", "image_1", "image_2", "candidate_source", "review_status"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> tuple[str, str]:
    return tuple(sorted((row["slug_1"], row["slug_2"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report remaining component pairs without rewriting queues")
    args = parser.parse_args()
    catalog = {r["Slug"]: r for r in read_csv(ROOT / "data/strapi/catalog_dataset.csv")}
    old = read_csv(ROOT / "data/near_duplicates/all_candidates.csv")
    decisions = [r for path in BATCHES.glob("*_decisions.csv") for r in read_csv(path)]
    legacy_exclusions = {
        r["review_id"] for r in decisions
        if r["source"] == "legacy_recheck" and r["verdict"] != "confirmed_near_duplicate"
    }
    confirmed = {key(r) for r in old if r["candidate_number"] not in legacy_exclusions}
    confirmed.update(key(r) for r in decisions if r["source"] != "legacy_recheck" and r["verdict"] == "confirmed_near_duplicate")
    reviewed = {key(r) for r in old}
    reviewed.update(key(r) for r in decisions)

    parent: dict[str, str] = {}

    def root(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in sorted(confirmed):
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        parent[root(a)] = root(b)
    groups: dict[str, list[str]] = defaultdict(list)
    for slug in parent:
        groups[root(slug)].append(slug)

    missing = sorted(
        pair for group in groups.values() for pair in itertools.combinations(sorted(group), 2)
        if pair not in reviewed
    )
    rows = []
    for a, b in missing:
        for slug in (a, b):
            if not catalog[slug]["Статус изображения"].startswith("ok_"):
                raise ValueError(f"unaccepted reference in confirmed component: {slug}")
        rid = hashlib.sha1((a + "\0" + b).encode("utf-8")).hexdigest()[:12]
        rows.append({
            "review_id": rid, "slug_1": a, "slug_2": b,
            "winery": catalog[a]["Винодельня"],
            "image_1": catalog[a]["Название фото"],
            "image_2": catalog[b]["Название фото"],
            "candidate_source": "component_closure", "review_status": "pending",
        })
    by_winery: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_winery[row["winery"]].append(row)
    parts = [[], [], []]
    for winery, group in sorted(by_winery.items(), key=lambda item: -len(item[1])):
        min(parts, key=len).extend(group)
    if not args.check:
        for n, part in enumerate(parts, 1):
            write_csv(BATCHES / f"closure_batch_{n}_queue.csv", sorted(part, key=lambda r: r["review_id"]))
    print(f"Confirmed edges: {len(confirmed)}; components: {len(groups)}; unreviewed component pairs: {len(rows)}")
    print(f"Closure batches: {[len(part) for part in parts]}")
    if args.check and rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
