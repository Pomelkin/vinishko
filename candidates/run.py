"""Run a versioned extractor on the fixed dataset and immediately score it."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from candidates.catalog import CATALOG_PATH, UNKNOWN  # noqa: E402
from candidates.eval import DATASET, evaluate_result, load_dataset, sha256  # noqa: E402


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def run(solution_name: str, dataset_path: Path = DATASET, limit: int | None = None) -> Path:
    if not re.fullmatch(r"v[0-9]+", solution_name):
        raise ValueError("Solution must be a version name such as v1")
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive")
    dataset_path = dataset_path.resolve()
    dataset = load_dataset(dataset_path)
    module = importlib.import_module(f"candidates.solution.{solution_name}.predictor")
    predict = module.predict
    solution_dir = HERE / "solution" / solution_name
    source_files = sorted(path for path in solution_dir.rglob("*") if path.is_file() and path.suffix in {".py", ".txt", ".json"})
    snapshot = {str(path.relative_to(HERE)).replace("\\", "/"): sha256(path) for path in source_files}
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    result_dir = RESULTS / f"{solution_name}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=False)
    metadata: dict[str, Any] = {
        "solution": solution_name,
        "started_at": datetime.now(UTC).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "catalog_sha256": sha256(CATALOG_PATH),
        "solution_files_sha256": snapshot,
        "pipeline_files_sha256": {
            str(path.relative_to(HERE)).replace("\\", "/"): sha256(path)
            for path in (HERE / "catalog.py", HERE / "match.py", HERE / "eval.py", HERE / "run.py")
        },
        "limit": limit,
    }
    (result_dir / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (result_dir / "predictions.jsonl").open("x", encoding="utf-8") as output:
        for gold in dataset[:limit]:
            started = time.perf_counter()
            try:
                prediction = dict(predict(ROOT / gold["image_path"]))
            except Exception as error:
                prediction = {
                    "category": UNKNOWN, "brand": UNKNOWN, "_status": "exception",
                    "_error": f"{type(error).__name__}: {error}",
                    "_trace": {"traceback": traceback.format_exc()},
                }
            row = {
                "id": gold["id"], "image_path": gold["image_path"],
                "prediction": prediction,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(f"{gold['id']}: {prediction.get('_status')} | {row['latency_ms']} ms", flush=True)
    metadata["finished_at"] = datetime.now(UTC).isoformat()
    (result_dir / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = evaluate_result(result_dir, dataset_path)
    print(json.dumps({"result_dir": str(result_dir), "rates": report["rates"],
                      "latency_ms": report["latency_ms"],
                      "candidate_count": report["candidate_count"]}, ensure_ascii=False, indent=2))
    return result_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", default="v1")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.solution, args.dataset, args.limit)


if __name__ == "__main__":
    main()
