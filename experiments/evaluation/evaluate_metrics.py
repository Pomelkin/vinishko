from __future__ import annotations

import csv
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "sorted_photos_with_retrieval.csv"

K = 3
TARGET_COLUMN = "slug"
RESULT_COLUMN = "retrieval_results"


def find_target_rank(row: dict[str, str]) -> int | None:
    results = json.loads(row[RESULT_COLUMN])
    target = str(row[TARGET_COLUMN])

    for rank, result in enumerate(results[:K], start=1):
        if result.get("metadata", {}).get("slug") == target:
            return rank
    return None


def main() -> None:
    if K < 1:
        raise ValueError("K должен быть положительным")

    with INPUT_CSV.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = {TARGET_COLUMN, RESULT_COLUMN} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"В датасете нет колонок: {sorted(missing)}")
        dataset = list(reader)

    if not dataset:
        raise ValueError("Датасет пуст")

    ranks = [find_target_rank(row) for row in dataset]
    recall = sum(rank is not None for rank in ranks) / len(ranks)
    mean_average_precision = sum(1.0 / rank for rank in ranks if rank is not None) / len(ranks)
    ndcg = sum(1.0 / math.log2(rank + 1) for rank in ranks if rank is not None) / len(ranks)

    print(f"Запросов: {len(dataset)}")
    print(f"Recall@{K}: {recall:.4f}")
    print(f"MAP@{K}:    {mean_average_precision:.4f}")
    print(f"NDCG@{K}:   {ndcg:.4f}")


if __name__ == "__main__":
    main()
