#!/usr/bin/env python3
"""Check that every generated near-duplicate candidate has one reviewed verdict."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BATCHES = ROOT / "data" / "near_duplicates" / "review_batches"
VERDICTS = {
    "confirmed_near_duplicate", "same_object", "same_wine_redesign",
    "data_collision", "visually_distinct", "needs_review",
    "needs_correct_reference",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def inputs() -> dict[tuple[str, str], dict[str, str]]:
    expected: dict[tuple[str, str], dict[str, str]] = {}
    paths = [
        *((BATCHES / f"batch_{n}_queue.csv", "queue", "review_id") for n in (1, 2, 3)),
        *((BATCHES / f"batch_{n}_required.csv", "required", "candidate_number") for n in (1, 2, 3)),
        *((BATCHES / f"batch_{n}_extra_075.csv", "extra_075", "review_id") for n in (1, 2, 3)),
        (BATCHES / "cross_winery_queue.csv", "cross_winery", "review_id"),
        (BATCHES / "old_gallery_mismatch_recheck.csv", "legacy_recheck", "review_id"),
        (BATCHES / "old_low_score_recheck.csv", "legacy_recheck", "review_id"),
        (BATCHES / "sweep_queue.csv", "sweep", "review_id"),
        *((BATCHES / f"closure_batch_{n}_queue.csv", "closure", "review_id") for n in (1, 2, 3)),
    ]
    for path, source, id_field in paths:
        if not path.is_file():
            continue
        for row in read_csv(path):
            key = source, row[id_field]
            if key in expected:
                if source == "legacy_recheck" and (expected[key]["slug_1"], expected[key]["slug_2"]) == (row["slug_1"], row["slug_2"]):
                    continue
                raise ValueError(f"input candidate repeated: {key}")
            expected[key] = row
    return expected


def main() -> int:
    expected = inputs()
    decisions: dict[tuple[str, str], dict[str, str]] = {}
    errors = []
    for path in sorted(BATCHES.glob("*_decisions.csv")):
        for row in read_csv(path):
            key = row.get("source", ""), row.get("review_id", "")
            if key in decisions:
                errors.append(f"duplicate decision {key} in {path.name}")
            decisions[key] = row
            inp = expected.get(key)
            if inp is None:
                errors.append(f"decision absent from input: {key} in {path.name}")
                continue
            if (row.get("slug_1"), row.get("slug_2")) != (inp["slug_1"], inp["slug_2"]):
                errors.append(f"slug mismatch: {key}")
            if row.get("verdict") not in VERDICTS:
                errors.append(f"invalid verdict: {key}: {row.get('verdict')}")
            if row.get("visual_checked", "").lower() == "true" and not row.get("evidence", "").strip():
                errors.append(f"reviewed row lacks evidence: {key}")
            if row.get("verdict") == "confirmed_near_duplicate" and row.get("visual_checked", "").lower() != "true":
                errors.append(f"unseen pair marked confirmed: {key}")
    missing = expected.keys() - decisions.keys()
    unchecked = {
        key for key, row in decisions.items()
        if key in expected and row.get("visual_checked", "").lower() != "true"
    }
    totals = Counter(row.get("verdict", "") for key, row in decisions.items() if key in expected and key not in unchecked)
    print(f"Input candidates: {len(expected)}; decision rows: {len(decisions)}; missing: {len(missing)}; unchecked: {len(unchecked)}")
    print(f"Reviewed verdicts: {dict(totals)}")
    current_audit = BATCHES / "current_audit_075_candidates.csv"
    if current_audit.is_file():
        reviewed_pairs = {frozenset((row["slug_1"], row["slug_2"])) for row in decisions.values()}
        audit_rows = read_csv(current_audit)
        unseen_audit = [row for row in audit_rows if frozenset((row["slug_1"], row["slug_2"])) not in reviewed_pairs]
        print(f"Current 0.75 audit non-registry candidates: {len(audit_rows)}; without review: {len(unseen_audit)}")
        for row in unseen_audit[:10]:
            errors.append(f"current 0.75 audit candidate lacks review: {row['slug_1']} | {row['slug_2']}")
    for error in errors[:30]:
        print(f"ERROR: {error}", file=sys.stderr)
    if len(errors) > 30:
        print(f"... {len(errors) - 30} more errors", file=sys.stderr)
    return 1 if errors or missing or unchecked else 0


if __name__ == "__main__":
    raise SystemExit(main())
