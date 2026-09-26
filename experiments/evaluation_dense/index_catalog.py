from __future__ import annotations

import csv
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client import models
from tqdm import tqdm

from siglip_encoder import Siglip2Encoder

HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
CATALOG_CSV = EXPERIMENTS_DIR / "data" / "catalog_dataset.csv"
IMAGES_DIR = EXPERIMENTS_DIR / "data" / "images_without_garbarage"

QDRANT_URL = "http://127.0.0.1:6333"
QDRANT_GRPC_PORT = 6334
COLLECTION_NAME = "catalog_siglip2_dense"
REQUEST_TIMEOUT_SECONDS = 600
ENCODE_BATCH_SIZE = 8
QDRANT_BATCH_SIZE = 64


def load_catalog() -> list[dict[str, str]]:
    with CATALOG_CSV.open(encoding="utf-8-sig", newline="") as file:
        rows = [
            {"slug": row["Slug"].strip(), "photo": row["Название фото"].strip()}
            for row in csv.DictReader(file)
            if row["Slug"].strip() and row["Название фото"].strip()
        ]

    missing = [row["photo"] for row in rows if not (IMAGES_DIR / row["photo"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Не найдены {len(missing)} изображений, первые: {missing[:5]}")
    if not rows:
        raise ValueError("В CSV нет строк с заполненными Slug и Название фото")
    return rows


def recreate_collection(client: QdrantClient, vector_size: int) -> None:
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )


def upload_batch(
        client: QdrantClient,
        rows: list[dict[str, str]],
        embeddings: list[list[float]],
        start_id: int,
) -> None:
    client.upload_points(
        collection_name=COLLECTION_NAME,
        batch_size=QDRANT_BATCH_SIZE,
        max_retries=3,
        wait=True,
        points=[
            models.PointStruct(
                id=start_id + offset,
                vector=embedding,
                payload={"slug": row["slug"], "photo": row["photo"]},
            )
            for offset, (row, embedding) in enumerate(zip(rows, embeddings, strict=True))
        ],
    )


def main() -> None:
    rows = load_catalog()
    encoder = Siglip2Encoder()
    first_batch = rows[:ENCODE_BATCH_SIZE]
    first_embeddings = encoder.encode([IMAGES_DIR / row["photo"] for row in first_batch])
    vector_size = len(first_embeddings[0])

    qdrant = QdrantClient(
        url=QDRANT_URL,
        grpc_port=QDRANT_GRPC_PORT,
        prefer_grpc=True,
        timeout=REQUEST_TIMEOUT_SECONDS,
        grpc_options={
            "grpc.max_send_message_length": -1,
            "grpc.max_receive_message_length": -1,
        },
    )
    try:
        recreate_collection(qdrant, vector_size)
        with tqdm(total=len(rows), desc="Индексирование SigLIP2", unit="изобр.") as progress:
            upload_batch(qdrant, first_batch, first_embeddings, 0)
            progress.update(len(first_batch))
            for start in range(ENCODE_BATCH_SIZE, len(rows), ENCODE_BATCH_SIZE):
                batch = rows[start: start + ENCODE_BATCH_SIZE]
                embeddings = encoder.encode([IMAGES_DIR / row["photo"] for row in batch])
                upload_batch(qdrant, batch, embeddings, start)
                progress.update(len(batch))

        count = qdrant.count(COLLECTION_NAME, exact=True).count
        if count != len(rows):
            raise RuntimeError(f"В Qdrant {count} точек вместо {len(rows)}")
        print(f"Готово: {count} dense-точек в коллекции {COLLECTION_NAME!r}, размерность {vector_size}")
    finally:
        qdrant.close()


if __name__ == "__main__":
    main()
