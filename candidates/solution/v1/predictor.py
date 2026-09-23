"""Single-image OpenRouter call with a strict catalog-backed output contract."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from candidates.catalog import ROOT, UNKNOWN, catalog_values


HERE = Path(__file__).resolve().parent


def _mime(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    raise ValueError("Неподдерживаемая сигнатура изображения")


def _settings() -> dict[str, Any]:
    settings = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    if settings["generation"].get("reasoning", {}).get("effort") != "none":
        raise ValueError("v1 must use reasoning effort none")
    return settings


def _schema(categories: tuple[str, ...], brands: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "wine_fallback_features",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": [*categories, UNKNOWN]},
                    "brand": {"type": "string", "enum": [*brands, UNKNOWN]},
                },
                "required": ["category", "brand"],
                "additionalProperties": False,
            },
        },
    }


def _validate(content: str, categories: tuple[str, ...], brands: tuple[str, ...]) -> dict[str, str]:
    value = json.loads(content)
    if not isinstance(value, dict) or set(value) != {"category", "brand"}:
        raise ValueError("Ответ должен содержать только category и brand")
    if value["category"] not in (*categories, UNKNOWN):
        raise ValueError("Категория вне Каталога")
    if value["brand"] not in (*brands, UNKNOWN):
        raise ValueError("Винодельня вне Каталога")
    return value


def _content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Ответ API не содержит choices")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ответ API не содержит текста")
    return content


def predict(image_path: str | Path) -> dict[str, Any]:
    """Minimal runner interface: one image path in, one validated prediction out.

    Operational failures retain an error status and never masquerade as abstention.
    """
    settings = _settings()
    categories, brands = catalog_values()
    data = Path(image_path).read_bytes()
    media_type = _mime(data)
    load_dotenv(ROOT / ".env", override=False)
    api_key = os.getenv(settings["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Не задан {settings['api_key_env']}")

    prompt = (HERE / "prompt.txt").read_text(encoding="utf-8").strip()
    options = (
        "Категории Каталога: " + json.dumps(categories, ensure_ascii=False)
        + "\nВинодельни Каталога: " + json.dumps(brands, ensure_ascii=False)
        + "\nДопустимый отказ для каждого поля: " + UNKNOWN
    )
    schema = _schema(categories, brands)
    # Include the schema in plain text too: some vision providers ignore metadata.
    system = prompt + "\n\n" + options + "\n\nJSON schema: " + json.dumps(
        schema["json_schema"]["schema"], ensure_ascii=False
    )
    payload = {
        "model": settings["model"],
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Определи категорию и винодельню основной бутылки."},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}",
                        "detail": settings["image_detail"],
                    }},
                ],
            },
        ],
        "response_format": schema,
        **settings["generation"],
        "provider": settings["provider"],
    }
    endpoint = settings["api_base"].rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "vinishko-candidates-v1/1",
    }
    if referer := os.getenv("OPENROUTER_SITE_URL"):
        headers["HTTP-Referer"] = referer
    if title := os.getenv("OPENROUTER_APP_TITLE"):
        headers["X-Title"] = title
    request = urllib.request.Request(
        endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers, method="POST",
    )
    trace = {
        "model": settings["model"],
        "image_sha256": hashlib.sha256(data).hexdigest(),
        "image_media_type": media_type,
        "prompt_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
        "raw_response": None,
    }
    try:
        with urllib.request.urlopen(request, timeout=settings["timeout_seconds"]) as response:
            raw = response.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw)
        trace["raw_response"] = parsed
        content = _content(parsed)
        try:
            prediction = _validate(content, categories, brands)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            return {"category": UNKNOWN, "brand": UNKNOWN,
                    "_status": "invalid_response", "_error": str(error), "_trace": trace}
        return {**prediction, "_status": "ok", "_error": None, "_trace": trace}
    except urllib.error.HTTPError as error:
        trace["http_status"] = error.code
        trace["raw_response_text"] = error.read().decode("utf-8", errors="replace")
        return {"category": UNKNOWN, "brand": UNKNOWN,
                "_status": "api_error", "_error": f"HTTP {error.code}", "_trace": trace}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        return {"category": UNKNOWN, "brand": UNKNOWN,
                "_status": "api_error", "_error": f"{type(error).__name__}: {error}", "_trace": trace}
