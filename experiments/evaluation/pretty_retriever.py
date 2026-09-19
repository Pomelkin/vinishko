from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from types import TracebackType
from typing import Any
from typing import Literal

import httpx
import httpx2
from openai import OpenAI
from pydantic import BaseModel
from pydantic import Field
from pydantic import create_model
from qdrant_client import QdrantClient

CATALOG_IMAGES_DIR = Path(__file__).parents[1] / "data" / "images_without_garbarage"
VLM_BASE_URL = "https://0e8d-34-158-148-54.ngrok-free.app/v1"
VLM_API_KEY = "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"
# VLM_MODEL = "qwen/qwen3.6-35b-a3b"
VLM_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
VLM_EXTRA_BODY: dict[str, Any] = {
    "chat_template_kwargs": {"enable_thinking": False},
    # "reasoning": {
    #   "enabled": False
    # },
    # "provider": {
    #     "only": ["siliconflow/fp8"]
    # }
}


def make_result_model(candidate_count: int) -> type[BaseModel]:
    allowed_numbers = Literal.__getitem__(tuple(range(1, candidate_count + 1)))
    pydantic_model = create_model(
        "WineMatch",
        image_number=(
            allowed_numbers,
            Field(description="Number of the single best matching candidate image."),
        ),
    )
    return pydantic_model


def build_system_prompt(result_model: type[BaseModel], candidate_count: int) -> str:
    schema = json.dumps(result_model.model_json_schema(), ensure_ascii=False)
    return f"""
You are a highly precise wine product matcher. Compare the query photo with
{candidate_count} numbered catalog candidate images and select exactly one.

Treat a candidate as an exact match only when the wine identity on the label
matches: producer, brand, product/cuvee/series name, vintage/year, grape variety,
color, sweetness, sparkling status, reserve designation, appellation, and other
visible identifying text. Pay special attention to the label. Wines from the
same producer or series are different products when any of those details differ.
Ignore glare, blur, perspective, background, lighting, bottle angle, and minor
packaging redesigns that do not change the product identity.

Return the number of the best exact match. If no candidate is certainly exact,
still return the single closest candidate. Do not explain the choice. Return
only JSON matching this schema:
{schema}
""".strip()


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
            use_vlm: bool = False,
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
        httpx_client = httpx2.Client(proxy="http://localhost:12334")
        self.vlm = (
            OpenAI(
                base_url=VLM_BASE_URL,
                api_key=VLM_API_KEY,
                timeout=timeout_seconds,
                max_retries=0,
                http_client=httpx_client,
            )
            if use_vlm
            else None
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

        results = [
            {
                "point_id": str(point.id),
                "metadata": dict(point.payload or {}),
                "raw_score": float(point.score),
                "normalized_score": float(point.score) / len(embedding),
            }
            for point in points
        ]
        return self._select_with_vlm(Path(image_path), results) if self.vlm else results

    def _select_with_vlm(
            self,
            query_image: Path,
            results: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if not results:
            return []
        if self.vlm is None:
            return results

        content: list[dict[str, Any]] = [
            {"type": "text", "text": "QUERY PHOTO:"},
            {
                "type": "image_url",
                "image_url": {"url": self._image_url(query_image), "detail": "high"},
            },
        ]
        for number, result in enumerate(results, start=1):
            metadata = result["metadata"]
            if not isinstance(metadata, dict) or "photo" not in metadata:
                raise ValueError("У кандидата нет metadata.photo")
            content.extend(
                [
                    {"type": "text", "text": f"CANDIDATE {number}:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._image_url(CATALOG_IMAGES_DIR / str(metadata["photo"])),
                        },
                    },
                ]
            )
        content.append(
            {
                "type": "text",
                "text": f"Return exactly one candidate number from 1 to {len(results)}.",
            }
        )

        result_model = make_result_model(len(results))
        completion = self.vlm.chat.completions.parse(
            model=VLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": build_system_prompt(result_model, len(results)),
                },
                {"role": "user", "content": content},
            ],
            response_format=result_model,
            temperature=0,
            extra_body=VLM_EXTRA_BODY,
        )
        message = completion.choices[0].message
        if message.parsed is None:
            raise RuntimeError(message.refusal or message.content or "VLM не вернула результат")

        index = message.parsed.model_dump()["image_number"] - 1
        return [results[index]]

    @staticmethod
    def _image_url(image_path: Path) -> str:
        if not image_path.is_file():
            raise FileNotFoundError(f"Не найдена фотография для VLM: {image_path}")
        media_type = mimetypes.guess_type(image_path.name)[0]
        if media_type is None and image_path.suffix.lower() == ".webp":
            media_type = "image/webp"
        if media_type is None or not media_type.startswith("image/"):
            raise ValueError(f"Неподдерживаемый формат изображения: {image_path}")
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{media_type};base64,{encoded}"

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
        if self.vlm:
            self.vlm.close()

    def __enter__(self) -> PrettyRetriever:
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
    ) -> None:
        self.close()
