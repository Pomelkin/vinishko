"""Классификация ближайших кандидатов через OpenAI-совместимый structured output."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import Any

from openai import OpenAI

from .config import VanillaVlmSettings
from .models import nearest_output_model

CARD_FIELDS = (
    "slug", "name", "winery", "vintage", "grapes", "category",
    "sugar", "sparkling", "abv", "aging_or_reserve",
)


def image_url(data: bytes) -> str:
    """Кодировать эталон в data URL с типом по сигнатуре."""
    if data.startswith(b"\xff\xd8\xff"):
        media_type = "image/jpeg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        media_type = "image/png"
    elif len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        media_type = "image/webp"
    else:
        raise ValueError("неподдерживаемый формат изображения для VLM")
    return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"


def classify(
    client: OpenAI,
    settings: VanillaVlmSettings,
    query_image: bytes,
    candidates: list[dict[str, Any]],
    prompts: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Выбрать slug из переданных кандидатов или not_found."""
    slugs = [candidate["slug"] for candidate in candidates]
    output_model = nearest_output_model(slugs)
    policies = []
    for number, candidate in enumerate(candidates, 1):
        vintage = str(candidate.get("vintage", "")).strip()
        policies.append((number, candidate["slug"], vintage))
    prompt_parts = [prompts["resolve_multiple_same"].rstrip()]
    if any(vintage for _, _, vintage in policies):
        prompt_parts.append(prompts["compare_year_matters"].strip())
    if any(not vintage for _, _, vintage in policies):
        prompt_parts.append(prompts["compare_year_not_matter"].strip())
    policy_lines = ["Vintage policy for every supplied element:"]
    for number, slug, vintage in policies:
        rule = f"an exact match with {vintage} is required" if vintage else "an exact vintage match is not required"
        policy_lines.append(f"- ELEMENT {number} ({slug}): {rule}.")
    prompt_parts.append("\n".join(policy_lines))
    prompt_parts.append("JSON_SCHEMA:\n" + json.dumps(output_model.model_json_schema(), ensure_ascii=False))

    content: list[dict[str, Any]] = [
        {"type": "text", "text": "=== QUERY ==="},
        {"type": "image_url", "image_url": {"url": image_url(query_image), "detail": "original"}},
    ]
    for number, candidate in enumerate(candidates, 1):
        card = {field: candidate.get(field, "") for field in CARD_FIELDS}
        content.extend([
            {"type": "text", "text": f"=== ELEMENT {number} ===\nCARD_JSON:\n{json.dumps(card, ensure_ascii=False)}"},
            {"type": "image_url", "image_url": {"url": image_url(candidate["reference_image_bytes"]), "detail": "original"}},
        ])
    content.append({"type": "text", "text": "Select one exact candidate slug, or not_found if all candidates have a confident product-identity conflict. Return only JSON matching the schema."})

    completion = client.chat.completions.parse(
        model=settings.model,
        messages=[
            {"role": "system", "content": "\n\n".join(prompt_parts)},
            {"role": "user", "content": content},
        ],
        response_format=output_model,
        temperature=0,
        max_completion_tokens=settings.max_completion_tokens,
        extra_body=settings.extra_body,
    )
    if not completion.choices:
        raise ValueError("VLM вернула пустой список ответов")
    message = completion.choices[0].message
    if message.parsed is None:
        raise ValueError(message.refusal or message.content or "VLM не вернула structured output")
    selected = message.parsed.model_dump(mode="json")
    trace = {
        "model": settings.model,
        "candidate_slugs": slugs,
        "response_id": completion.id,
        "selected": selected,
    }
    return selected, trace
