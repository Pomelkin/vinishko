from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any
from typing import Literal

from openai import DefaultHttpxClient
from openai import OpenAI
from pydantic import BaseModel
from pydantic import Field
from pydantic import create_model

# Настройки запуска.
VLLM_BASE_URL = "https://5214-34-147-86-224.ngrok-free.app/v1"
VLLM_API_KEY = "EMPTY"
MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
QUERY_IMAGE = r"E:\Pycharm Projects\vinishko\experiments\photo_viewer\sorted_photos\abrau-dyurso-russkoe-igristoe-polusladkoe-shardone-beloe-12\images.jpg"
CANDIDATES_DIR = Path(__file__).with_name("candidates")
REQUEST_TIMEOUT_SECONDS = 600.0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VLLM_EXTRA_BODY: dict[str, Any] = {
    "chat_template_kwargs": {"enable_thinking": False},
}


def image_to_data_url(image_path: str | Path) -> str:
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    media_type = mimetypes.guess_type(image_path.name)[0]
    if media_type is None and image_path.suffix.lower() == ".webp":
        media_type = "image/webp"
    if media_type is None or not media_type.startswith("image/"):
        raise ValueError(f"Unsupported image type: {image_path}")

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def get_candidates() -> list[Path]:
    if not CANDIDATES_DIR.is_dir():
        raise FileNotFoundError(f"Candidates directory not found: {CANDIDATES_DIR}")

    candidates = sorted(
        (
            path
            for path in CANDIDATES_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )
    if not candidates:
        raise ValueError(f"No images found in: {CANDIDATES_DIR}")
    return candidates


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


def build_user_content(
        query_image: str | Path,
        candidates: list[Path],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "QUERY PHOTO:"},
        {
            "type": "image_url",
            "image_url": {"url": image_to_data_url(query_image), "detail": "high"},
        },
    ]
    for number, candidate in enumerate(candidates, start=1):
        print(f"Adding candidate {number}: {candidate}")
        content.extend(
            [
                {"type": "text", "text": f"CANDIDATE {number}:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_data_url(candidate),
                        # "detail": "high",
                    },
                },
            ]
        )
    content.append(
        {
            "type": "text",
            "text": f"Return exactly one candidate number from 1 to {len(candidates)}.",
        }
    )
    return content


def main() -> None:
    candidates = get_candidates()
    result_model = make_result_model(len(candidates))
    messages: list[Any] = [
        {
            "role": "system",
            "content": build_system_prompt(result_model, len(candidates)),
        },
        {
            "role": "user",
            "content": build_user_content(QUERY_IMAGE, candidates),
        },
    ]

    with OpenAI(
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
            http_client=DefaultHttpxClient(trust_env=False),
    ) as client:
        completion = client.chat.completions.parse(
            model=MODEL,
            messages=messages,
            response_format=result_model,
            temperature=0,
            extra_body=VLLM_EXTRA_BODY,
        )

    message = completion.choices[0].message
    if message.parsed is None:
        raise RuntimeError(message.refusal or message.content or "Model returned no result")
    image_number = message.parsed.model_dump()["image_number"]
    print(candidates[image_number - 1].name)


if __name__ == "__main__":
    main()
