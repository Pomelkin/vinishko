from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pretty_retriever import PrettyRetriever


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]

INPUT_CSV = HERE / "sorted_photos.csv"
OUTPUT_CSV = HERE / "sorted_photos_with_retrieval.csv"
IMAGE_COLUMN = "image"
RESULT_COLUMN = "retrieval_results"
TOP_K = 10

ENCODER_URL = "http://localhost:8000/encode"
QDRANT_URL = "http://127.0.0.1:6333"
QDRANT_GRPC_PORT = 6334
COLLECTION_NAME = "catalog_evie"
REQUEST_TIMEOUT_SECONDS = 600


def main() -> None:
    dataset: pd.DataFrame = pd.read_csv(str(INPUT_CSV))
    if IMAGE_COLUMN not in dataset.columns:
        raise ValueError(f"В датасете нет колонки {IMAGE_COLUMN!r}")

    with PrettyRetriever(
        encoder_url=ENCODER_URL,
        qdrant_url=QDRANT_URL,
        qdrant_grpc_port=QDRANT_GRPC_PORT,
        collection_name=COLLECTION_NAME,
        timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    ) as retriever:
        results = dataset[IMAGE_COLUMN].apply(
            lambda image: json.dumps(
                retriever.retrieve(PROJECT_DIR / str(image), limit=TOP_K),
                ensure_ascii=False,
            )
        )

    dataset.insert(dataset.columns.get_loc(IMAGE_COLUMN) + 1, RESULT_COLUMN, results)
    dataset.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Сохранено {len(dataset)} строк в {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
