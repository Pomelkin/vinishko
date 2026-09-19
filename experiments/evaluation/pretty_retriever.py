from __future__ import annotations

import mimetypes
from pathlib import Path
from types import TracebackType

import httpx
from qdrant_client import QdrantClient


class PrettyRetriever:
    """Encode one image and retrieve the closest catalog items from Qdrant."""

    def __init__(
        self,
        *,
        encoder_url: str,
        qdrant_url: str,
        qdrant_grpc_port: int,
        collection_name: str,
        timeout_seconds: int = 600,
    ) -> None:
        self.encoder_url = encoder_url
        self.collection_name = collection_name
        self.encoder = httpx.Client(timeout=timeout_seconds, trust_env=False)
        self.qdrant = QdrantClient(
            url=qdrant_url,
            grpc_port=qdrant_grpc_port,
            prefer_grpc=True,
            timeout=timeout_seconds,
            grpc_options={
                "grpc.max_send_message_length": -1,
                "grpc.max_receive_message_length": -1,
            },
        )

    def retrieve(self, image_path: str | Path, *, limit: int = 10) -> list[dict[str, object]]:
        """Return payload and raw/mean MaxSim score for the nearest points."""
        embedding = self._encode(Path(image_path))
        points = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=limit,
            with_payload=True,
        ).points

        return [
            {
                "point_id": str(point.id),
                "metadata": dict(point.payload or {}),
                "raw_score": float(point.score),
                "normalized_score": float(point.score) / len(embedding),
            }
            for point in points
        ]

    def _encode(self, image_path: Path) -> list[list[float]]:
        if not image_path.is_file():
            raise FileNotFoundError(f"Не найдена фотография: {image_path}")

        with image_path.open("rb") as image:
            response = self.encoder.post(
                self.encoder_url,
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
        body = response.json()
        embeddings = body.get("embeddings") if isinstance(body, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != 1 or not embeddings[0]:
            raise ValueError("Encoder вернул некорректный embedding")
        return embeddings[0]

    def close(self) -> None:
        self.encoder.close()
        self.qdrant.close()

    def __enter__(self) -> PrettyRetriever:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
