"""Оценка сохранённых прогонов vanilla_vlm по test.csv, без повторного запуска модели."""

import csv
import json
from pathlib import Path

TEST_CSV = Path("datasets/local/test/test.csv")
RUNS_DIR = Path("datasets/local/runs_vanilla")
KS = (1, 2, 3)


def unique(slugs: list[str]) -> list[str]:
    return list(dict.fromkeys(slugs))


def retrieval_slugs(run_dir: Path, result: dict) -> list[str]:
    search_file = run_dir / "search" / "results.json"
    queries = {}
    if search_file.is_file():
        queries = {query["query"]: query for query in json.loads(search_file.read_text(encoding="utf-8"))["queries"]}

    slugs = []
    for bottle in result["bottles"]:
        uuid = bottle["uuid"]
        trace_file = run_dir / "rerank" / f"{uuid}.json"
        if trace_file.is_file():
            trace = json.loads(trace_file.read_text(encoding="utf-8"))
            slugs.extend(trace["candidate_slugs"])
        elif (bottle.get("selection") or {}).get("source") == "vanilla_vlm":
            raise ValueError(f"Нет списка исходных кандидатов: {trace_file}")
        elif uuid in queries:
            if queries[uuid].get("selection"):
                raise ValueError(f"В {search_file} только результат реранка, исходный top-k неизвестен")
            slugs.extend(candidate["slug"] for candidate in queries[uuid]["candidates"])
    return unique(slugs)


def final_slugs(result: dict) -> list[str]:
    return unique([
        candidate["slug"]
        for bottle in result["bottles"]
        for candidate in bottle.get("candidates", [])
    ])


def scores(rows: list[tuple[str, list[str]]], k: int) -> tuple[float, float, float | None]:
    positives = [(slug, predicted) for slug, predicted in rows if slug]
    negatives = [predicted for slug, predicted in rows if not slug]
    ranks = [predicted[:k].index(slug) + 1 if slug in predicted[:k] else None for slug, predicted in positives]
    recall = sum(rank is not None for rank in ranks) / len(positives) if positives else 0.0
    mean_ap = sum(1 / rank for rank in ranks if rank is not None) / len(positives) if positives else 0.0
    rejection = sum(not predicted[:k] for predicted in negatives) / len(negatives) if negatives else None
    return recall, mean_ap, rejection


def main() -> None:
    with TEST_CSV.open(encoding="utf-8-sig", newline="") as file:
        truth = {row["image_filename"]: row["slug"].strip() for row in csv.DictReader(file)}

    retrieval = []
    final = []
    for index, (filename, slug) in enumerate(truth.items(), 1):
        run_dir = RUNS_DIR / Path(filename).stem
        if not (run_dir / "result.json").is_file():
            run_dir = RUNS_DIR / f"{index:05d}_{Path(filename).stem}"
        result_file = run_dir / "result.json"
        if not result_file.is_file():
            continue
        result = json.loads(result_file.read_text(encoding="utf-8"))
        retrieval.append((slug, retrieval_slugs(run_dir, result)))
        final.append((slug, final_slugs(result)))

    if not final:
        raise FileNotFoundError(f"в {RUNS_DIR} не найдено ни одного result.json для {TEST_CSV}")
    print(f"Оценено {len(final)}/{len(truth)} фото; отсутствующих прогонов: {len(truth) - len(final)}")
    for stage, rows in (("retrieval", retrieval), ("final", final)):
        print(f"{stage}: {sum(bool(slug) for slug, _ in rows)} positive, {sum(not slug for slug, _ in rows)} negative")
        for k in KS:
            recall, mean_ap, rejection = scores(rows, k)
            print(f"  @{k}: recall={recall:.4f} mAP={mean_ap:.4f} correct_reject={rejection:.4f}" if rejection is not None else f"  @{k}: recall={recall:.4f} mAP={mean_ap:.4f} correct_reject=n/a")


if __name__ == "__main__":
    main()
