from __future__ import annotations

import mimetypes
from pathlib import Path

import cv2
import httpx
import numpy as np
from qdrant_client import QdrantClient

HERE = Path(__file__).resolve().parent
IMAGES_DIR = HERE / "data" / "images_without_garbarage"

QUERY_IMAGE = Path(
    r"E:\Pycharm Projects\vinishko\experiments\data\images_without_garbarage\(-)_Аристов Кюве Розе_Пино_2023_0,75л_брют Normal-Photoroom.webp")
SHOW_IMAGES = True
TOP_K = 5

ENCODER_URL = "http://localhost:8000/encode"
QDRANT_URL = "http://127.0.0.1:6333"
QDRANT_GRPC_PORT = 6334
COLLECTION_NAME = "catalog_evie"
REQUEST_TIMEOUT_SECONDS = 600.0


def encode(image_path: Path) -> list[list[float]]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Не найдена фотография запроса: {image_path}")

    with image_path.open("rb") as image, httpx.Client(
            timeout=REQUEST_TIMEOUT_SECONDS,
            trust_env=False,
    ) as client:
        response = client.post(
            ENCODER_URL,
            files={
                "files": (
                    image_path.name,
                    image,
                    mimetypes.guess_type(image_path.name)[0] or "application/octet-stream",
                ),
            },
            headers={"ngrok-skip-browser-warning": "true"},
        )

    response.raise_for_status()
    embeddings = response.json()["embeddings"]
    if len(embeddings) != 1 or not embeddings[0]:
        raise ValueError("Encoder вернул некорректный embedding")
    return embeddings[0]


def show_images(results: list[tuple[str, str, float]]) -> None:
    for rank, (slug, photo, score) in enumerate(results, start=1):
        path = IMAGES_DIR / photo
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV не смог открыть {path}")

        title = f"#{rank} {slug} similarity={score:.2%}"
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    embedding = encode(QUERY_IMAGE)
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
        points = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=TOP_K,
            with_payload=True,
        ).points
    finally:
        qdrant.close()

    results = [
        (
            str(point.payload["slug"]),
            str(point.payload["photo"]),
            point.score / len(embedding),
        )
        for point in points
        if point.payload is not None
    ]
    for rank, (slug, _photo, score) in enumerate(results, start=1):
        print(f"{rank}. {slug}: {score:.2%}")

    if SHOW_IMAGES:
        show_images(results)


if __name__ == "__main__":
    main()
