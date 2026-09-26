from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from types import TracebackType
from typing import Any
from typing import Literal

import httpx
import tenacity
from openai import OpenAI
from pydantic import BaseModel
from pydantic import Field
from pydantic import create_model
from qdrant_client import QdrantClient

from siglip_encoder import Siglip2Encoder

HERE = Path(__file__).resolve().parent
CATALOG_IMAGES_DIR = HERE.parent / "data" / "images_without_garbarage"

QDRANT_URL = "http://127.0.0.1:6333"
QDRANT_GRPC_PORT = 6334
COLLECTION_NAME = "catalog_siglip2_dense"
REQUEST_TIMEOUT_SECONDS = 600

VLM_BASE_URL = "https://c3ed-35-197-132-174.ngrok-free.app/v1"
VLM_API_KEY = ""
VLM_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
VLM_PROXY: str | None = None
VLM_EXTRA_BODY: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}


def make_result_model(candidate_count: int) -> type[BaseModel]:
    allowed_numbers = Literal.__getitem__(tuple(range(1, candidate_count + 1)))
    return create_model(
        "WineMatch",
        image_number=(
            allowed_numbers,
            Field(description="Number of the single best matching candidate image."),
        ),
    )


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


class DenseRetriever:
    """Retrieve dense SigLIP2 image embeddings from Qdrant."""

    def __init__(self, *, use_vlm: bool = False) -> None:
        self.encoder = Siglip2Encoder()
        self.qdrant = QdrantClient(
            url=QDRANT_URL,
            grpc_port=QDRANT_GRPC_PORT,
            prefer_grpc=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
            grpc_options={
                "grpc.max_send_message_length": -1,
                "grpc.max_receive_message_length": -1,
            },
        )
        if use_vlm and not VLM_API_KEY:
            raise ValueError("Для USE_VLM=True заполните VLM_API_KEY в dense_retriever.py")
        self.vlm = (
            OpenAI(
                base_url=VLM_BASE_URL,
                api_key=VLM_API_KEY,
                timeout=REQUEST_TIMEOUT_SECONDS,
                max_retries=0,
                http_client=httpx.Client(proxy=VLM_PROXY, trust_env=False),
            )
            if use_vlm
            else None
        )

    def retrieve(self, image_path: str | Path, *, limit: int = 10) -> list[dict[str, object]]:
        _, selected = self.retrieve_with_candidates(image_path, limit=limit)
        return selected

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    def retrieve_with_candidates(
            self,
            image_path: str | Path,
            *,
            limit: int = 10,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        image_path = Path(image_path)
        embedding = self.encoder.encode([image_path])[0]
        points = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=limit,
            with_payload=True,
        ).points
        results = [
            {
                "point_id": str(point.id),
                "metadata": dict(point.payload or {}),
                "raw_score": float(point.score),
                "normalized_score": float(point.score),
            }
            for point in points
        ]
        selected = self._select_with_vlm(image_path, results) if self.vlm else results
        return results, selected

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
            {"type": "image_url", "image_url": {"url": self._image_url(query_image), "detail": "high"}},
        ]
        for number, result in enumerate(results, start=1):
            metadata = result["metadata"]
            if not isinstance(metadata, dict) or "photo" not in metadata:
                raise ValueError("У кандидата нет metadata.photo")
            content.extend([
                {"type": "text", "text": f"CANDIDATE {number}:"},
                {"type": "image_url",
                 "image_url": {"url": self._image_url(CATALOG_IMAGES_DIR / str(metadata["photo"]))}},
            ])
        content.append({"type": "text", "text": f"Return exactly one candidate number from 1 to {len(results)}."})

        result_model = make_result_model(len(results))
        completion = self.vlm.chat.completions.parse(
            model=VLM_MODEL,
            messages=[
                {"role": "system", "content": build_system_prompt(result_model, len(results))},
                {"role": "user", "content": content},
            ],
            response_format=result_model,
            temperature=0,
            extra_body=VLM_EXTRA_BODY,
        )
        message = completion.choices[0].message
        if message.parsed is None:
            raise RuntimeError(message.refusal or message.content or "VLM не вернула результат")
        return [results[message.parsed.model_dump()["image_number"] - 1]]

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

    def close(self) -> None:
        self.qdrant.close()
        if self.vlm:
            self.vlm.close()

    def __enter__(self) -> DenseRetriever:
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
    ) -> None:
        self.close()
