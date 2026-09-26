from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from dense_retriever import DenseRetriever

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]
INPUT_CSV = HERE.parent / "evaluation" / "sorted_photos.csv"
OUTPUT_CSV = HERE / "sorted_photos_with_retrieval_siglip2_dense.csv"
IMAGE_COLUMN = "image"
RESULT_COLUMN = "retrieval_results"
BEFORE_VLM_COLUMN = "retrieval_results_before_vlm"
TOP_K = 3
USE_VLM = False

tqdm.pandas(desc="SigLIP2 dense retrieval")


def main() -> None:
    dataset: pd.DataFrame = pd.read_csv(str(INPUT_CSV))
    if IMAGE_COLUMN not in dataset.columns:
        raise ValueError(f"В датасете нет колонки {IMAGE_COLUMN!r}")

    with DenseRetriever(use_vlm=USE_VLM) as retriever:
        retrievals = dataset[IMAGE_COLUMN].progress_apply(
            lambda image: retriever.retrieve_with_candidates(PROJECT_DIR / str(image), limit=TOP_K)
        )

    image_column_index = dataset.columns.get_loc(IMAGE_COLUMN)
    dataset.insert(
        image_column_index + 1,
        BEFORE_VLM_COLUMN,
        retrievals.map(lambda pair: json.dumps(pair[0], ensure_ascii=False)),
    )
    dataset.insert(
        image_column_index + 2,
        RESULT_COLUMN,
        retrievals.map(lambda pair: json.dumps(pair[1], ensure_ascii=False)),
    )
    dataset.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Сохранено {len(dataset)} строк в {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
