"""Offline, deterministic scoring of a candidates run; no model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from candidates.catalog import ROOT, UNKNOWN, catalog_values
from candidates.match import match_products


DATASET = Path(__file__).resolve().parent / "dataset.jsonl"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(path: Path = DATASET) -> list[dict[str, Any]]:
    categories, brands = catalog_values()
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or len({row["id"] for row in records}) != len(records):
        raise ValueError("Dataset empty or IDs are duplicated")
    for row in records:
        image_path = (ROOT / row["image_path"]).resolve()
        if not image_path.is_relative_to(ROOT) or not image_path.is_file():
            raise ValueError(f"Missing or unsafe image path: {row['image_path']}")
        if sha256(image_path) != row["image_sha256"]:
            raise ValueError(f"Image changed: {row['image_path']}")
        if row["category"] not in (*categories, UNKNOWN):
            raise ValueError(f"Unknown gold category: {row['id']}")
        if row["brand"] not in (*brands, UNKNOWN):
            raise ValueError(f"Unknown gold brand: {row['id']}")
    return records


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def evaluate(dataset: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    categories, brands = catalog_values()
    by_id: dict[str, dict[str, Any]] = {}
    for row in predictions:
        if row["id"] in by_id:
            raise ValueError(f"Duplicate prediction ID: {row['id']}")
        by_id[row["id"]] = row
    ids = {row["id"] for row in dataset}
    if set(by_id) - ids:
        raise ValueError(f"Predictions contain unknown IDs: {sorted(set(by_id) - ids)}")

    rows: list[dict[str, Any]] = []
    for gold in dataset:
        record = by_id.get(gold["id"])
        prediction = record.get("prediction", {}) if record else {}
        status = prediction.get("_status", "missing")
        valid = (status == "ok"
                 and prediction.get("category") in (*categories, UNKNOWN)
                 and prediction.get("brand") in (*brands, UNKNOWN))
        if status == "ok" and not valid:
            status = "invalid_prediction"
        category = prediction.get("category") if valid else None
        brand = prediction.get("brand") if valid else None
        category_ok = valid and category == gold["category"]
        brand_ok = valid and brand == gold["brand"]
        products = match_products(prediction) if valid else []
        slugs = {product["Slug"] for product in products}
        rows.append({
            "id": gold["id"], "image_path": gold["image_path"],
            "gold": {"category": gold["category"], "brand": gold["brand"]},
            "prediction": {"category": category, "brand": brand},
            "status": status, "category_correct": category_ok,
            "brand_correct": brand_ok, "both_correct": category_ok and brand_ok,
            "candidate_count": len(products),
            "source_slug_in_candidates": gold.get("source_slug") in slugs,
            "latency_ms": record.get("latency_ms") if record else None,
        })
    count = len(rows)
    latencies = sorted(row["latency_ms"] for row in rows if isinstance(row["latency_ms"], (int, float)))
    p95 = latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None
    candidate_counts = [row["candidate_count"] for row in rows]
    totals = {
        "dataset_size": count,
        "predicted": len(by_id),
        "valid": sum(row["status"] == "ok" for row in rows),
        "category_correct": sum(row["category_correct"] for row in rows),
        "brand_correct": sum(row["brand_correct"] for row in rows),
        "both_correct": sum(row["both_correct"] for row in rows),
        "source_slug_in_candidates": sum(row["source_slug_in_candidates"] for row in rows),
        "category_abstentions": sum(row["prediction"]["category"] == UNKNOWN for row in rows),
        "brand_abstentions": sum(row["prediction"]["brand"] == UNKNOWN for row in rows),
        "failures": sum(row["status"] not in {"ok", "missing"} for row in rows),
    }
    return {
        "totals": totals,
        "rates": {
            "coverage": _rate(totals["predicted"], count),
            "valid_rate": _rate(totals["valid"], count),
            "category_accuracy": _rate(totals["category_correct"], count),
            "brand_accuracy": _rate(totals["brand_correct"], count),
            "both_accuracy": _rate(totals["both_correct"], count),
            "source_slug_recall": _rate(totals["source_slug_in_candidates"], count),
            "category_abstention_rate": _rate(totals["category_abstentions"], count),
            "brand_abstention_rate": _rate(totals["brand_abstentions"], count),
        },
        "latency_ms": {
            "p50": round(statistics.median(latencies), 1) if latencies else None,
            "p95": round(p95, 1) if p95 is not None else None,
        },
        "candidate_count": {
            "p50": statistics.median(candidate_counts) if candidate_counts else None,
            "max": max(candidate_counts) if candidate_counts else None,
        },
        "cases": rows,
    }


def evaluate_result(result_dir: Path, dataset_path: Path = DATASET) -> dict[str, Any]:
    run = json.loads((result_dir / "run.json").read_text(encoding="utf-8"))
    if run["dataset_sha256"] != sha256(dataset_path):
        raise ValueError("Dataset hash differs from the run snapshot")
    catalog_hash = run.get("catalog_sha256")
    if catalog_hash is not None:
        from candidates.catalog import CATALOG_PATH
        if catalog_hash != sha256(CATALOG_PATH):
            raise ValueError("Catalog hash differs from the run snapshot")
    dataset = load_dataset(dataset_path)
    predictions = [json.loads(line) for line in (result_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    report = evaluate(dataset, predictions)
    report["run"] = run
    (result_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()
    report = evaluate_result(args.result_dir, args.dataset)
    print(json.dumps({"totals": report["totals"], "rates": report["rates"],
                      "latency_ms": report["latency_ms"],
                      "candidate_count": report["candidate_count"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
