from __future__ import annotations

import csv
import mimetypes
from contextlib import ExitStack
from pathlib import Path

import httpx
from qdrant_client import QdrantClient
from qdrant_client import models
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
CATALOG_CSV = HERE / "data" / "catalog_dataset.csv"
IMAGES_DIR = HERE / "data" / "images_without_garbarage"

ENCODER_URL = "http://localhost:8001/encode"
QDRANT_URL = "http://127.0.0.1:6333"
QDRANT_GRPC_PORT = 6334
COLLECTION_NAME = "catalog_neo"

BATCH_SIZE = 8
QDRANT_BATCH_SIZE = 1
REQUEST_TIMEOUT_SECONDS = 600.0
TEST_FILES_COUNT = 2


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


def encode(client: httpx.Client, rows: list[dict[str, str]]) -> list[list[list[float]]]:
    with ExitStack() as stack:
        files = [
            (
                "files",
                (
                    row["photo"],
                    stack.enter_context((IMAGES_DIR / row["photo"]).open("rb")),
                    mimetypes.guess_type(row["photo"])[0] or "application/octet-stream",
                ),
            )
            for row in rows
        ]
        response = client.post(
            ENCODER_URL,
            files=files,
            headers={"ngrok-skip-browser-warning": "true"},
        )
    response.raise_for_status()
    embeddings = response.json()["embeddings"]
    if len(embeddings) != len(rows) or any(not embedding for embedding in embeddings):
        raise ValueError("Encoder вернул некорректное число embeddings")
    return embeddings


def recreate_collection(client: QdrantClient, vector_size: int) -> None:
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
        ),
    )


def upload_batch(
    client: QdrantClient,
    rows: list[dict[str, str]],
    embeddings: list[list[list[float]]],
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


def point_count(client: QdrantClient) -> int:
    return client.count(COLLECTION_NAME, exact=True).count


def main() -> None:
    rows = load_catalog()
    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS)

    with httpx.Client(timeout=timeout, trust_env=False) as encoder:
        test_rows = rows[:TEST_FILES_COUNT]
        test_embeddings = encode(encoder, test_rows)
        vector_size = len(test_embeddings[0][0])
        if any(len(vector) != vector_size for embedding in test_embeddings for vector in embedding):
            raise ValueError("Encoder вернул vectors разной размерности")
        print(f"Тест API успешен: {len(test_rows)} файла, размерность {vector_size}")

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
            print(f"Коллекция {COLLECTION_NAME!r} пересоздана")

            with tqdm(total=len(rows), desc="Индексирование", unit="изобр.") as progress:
                for start in range(0, len(rows), BATCH_SIZE):
                    batch = rows[start : start + BATCH_SIZE]
                    embeddings = encode(encoder, batch)
                    upload_batch(qdrant, batch, embeddings, start)
                    progress.update(len(batch))

            count = point_count(qdrant)
            if count != len(rows):
                raise RuntimeError(f"В Qdrant {count} точек вместо {len(rows)}")
            print(f"Готово: {count} точек в коллекции {COLLECTION_NAME!r}")
        finally:
            qdrant.close()


if __name__ == "__main__":
    main()
