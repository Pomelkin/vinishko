"""Three-stage OpenRouter-compatible predictor for the NDR prompt suite."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

try:
    from .models import ModelContractError
    from .models import comparison_output_model
    from .models import resolver_output_model
    from .models import response_format
    from .models import validate_output
except ImportError:  # Loaded directly by ndr/run/run.py.
    from models import ModelContractError
    from models import comparison_output_model
    from models import resolver_output_model
    from models import response_format
    from models import validate_output


NOT_FOUND = "not_found"
CARD_FIELDS = (
    "slug",
    "name",
    "winery",
    "vintage",
    "grapes",
    "category",
    "sugar",
    "sparkling",
    "abv",
    "aging_or_reserve",
)


def image_media_type(data: bytes) -> str:
    """Detect a supported image media type from bytes, never from a filename."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("unsupported image signature")


def image_blocks(
    path_value: str,
    detail: str = "original",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an API image block and a base64-free trace block."""
    path = Path(path_value)
    data = path.read_bytes()
    media_type = image_media_type(data)
    api_block = {
        "type": "image_url",
        "image_url": {
            "url": (
                f"data:{media_type};base64,"
                f"{base64.b64encode(data).decode('ascii')}"
            ),
            "detail": detail,
        },
    }
    trace_block = {
        "type": "image_url",
        "image": {
            "path": str(path),
            "media_type": media_type,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "detail": detail,
            "data_url_omitted": True,
        },
    }
    return api_block, trace_block


def candidate_card(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Select compact, label-relevant catalog fields."""
    return {field: candidate.get(field, "") for field in CARD_FIELDS}


def add_text(
    api_content: list[dict[str, Any]],
    trace_content: list[dict[str, Any]],
    text: str,
) -> None:
    """Append the same text content block to API and trace messages."""
    block = {"type": "text", "text": text}
    api_content.append(block)
    trace_content.append(dict(block))


def add_image(
    api_content: list[dict[str, Any]],
    trace_content: list[dict[str, Any]],
    path: str,
    detail: str,
) -> None:
    """Append an encoded API image and a compact trace descriptor."""
    api_block, trace_block = image_blocks(path, detail)
    api_content.append(api_block)
    trace_content.append(trace_block)


def comparison_content(
    query_path: str,
    candidate: Mapping[str, Any],
    image_detail: str = "original",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the QUERY-versus-one-ELEMENT multimodal message."""
    api_content: list[dict[str, Any]] = []
    trace_content: list[dict[str, Any]] = []
    add_text(api_content, trace_content, "=== QUERY ===")
    add_image(api_content, trace_content, query_path, image_detail)
    card_json = json.dumps(candidate_card(candidate), ensure_ascii=False)
    add_text(
        api_content,
        trace_content,
        f"=== ELEMENT ===\nCARD_JSON:\n{card_json}",
    )
    add_image(
        api_content,
        trace_content,
        candidate["reference_image_path"],
        image_detail,
    )
    add_text(
        api_content,
        trace_content,
        "Сравни QUERY и ELEMENT. Верни только JSON по structured output schema.",
    )
    return api_content, trace_content


def resolution_content(
    query_path: str,
    candidates: list[Mapping[str, Any]],
    image_detail: str = "original",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the QUERY-versus-multiple-matches resolver message."""
    api_content: list[dict[str, Any]] = []
    trace_content: list[dict[str, Any]] = []
    add_text(api_content, trace_content, "=== QUERY ===")
    add_image(api_content, trace_content, query_path, image_detail)
    for number, candidate in enumerate(candidates, start=1):
        card_json = json.dumps(candidate_card(candidate), ensure_ascii=False)
        add_text(
            api_content,
            trace_content,
            f"=== ELEMENT {number} ===\nCARD_JSON:\n{card_json}",
        )
        add_image(
            api_content,
            trace_content,
            candidate["reference_image_path"],
            image_detail,
        )
    add_text(
        api_content,
        trace_content,
        "Выбери ровно один ELEMENT. Верни только JSON по structured output schema.",
    )
    return api_content, trace_content


def content_text(content: Any) -> str | None:
    """Extract text from a Chat Completions message content value."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            block.get("text", "")
            for block in content
            if isinstance(block, Mapping) and isinstance(block.get("text"), str)
        ]
        return "".join(texts) if texts else None
    return None


def generation_records(
    raw_response: Mapping[str, Any],
    validator: Callable[[Any], dict[str, Any]],
) -> tuple[list[dict[str, Any]], int | None, dict[str, Any] | None]:
    """Preserve and validate every provider choice without discarding reasoning."""
    records = []
    selected_index = None
    selected = None
    choices = raw_response.get("choices", [])
    if not isinstance(choices, list):
        choices = []
    for ordinal, choice in enumerate(choices):
        choice_mapping = choice if isinstance(choice, Mapping) else {}
        message_value = choice_mapping.get("message", {})
        message = message_value if isinstance(message_value, Mapping) else {}
        text = content_text(message.get("content"))
        parsed = None
        validated = None
        validation_error = None
        if text is None:
            validation_error = "generation has no textual content"
        else:
            try:
                parsed = json.loads(text)
                validated = validator(parsed)
            except (json.JSONDecodeError, ModelContractError) as error:
                validation_error = f"{type(error).__name__}: {error}"
        records.append(
            {
                "ordinal": ordinal,
                "provider_index": choice_mapping.get("index", ordinal),
                "finish_reason": choice_mapping.get("finish_reason"),
                "content": message.get("content"),
                "reasoning": message.get("reasoning"),
                "reasoning_details": message.get("reasoning_details"),
                "refusal": message.get("refusal"),
                "annotations": message.get("annotations"),
                "parsed": parsed,
                "validated": validated,
                "validation_error": validation_error,
                "raw_choice": choice,
            },
        )
        if selected is None and validated is not None:
            selected_index = ordinal
            selected = validated
    return records, selected_index, selected


def endpoint_url(api_base: str) -> str:
    """Normalize an API base into a chat-completions URL."""
    normalized = api_base.rstrip("/")
    return normalized if normalized.endswith("/chat/completions") else f"{normalized}/chat/completions"


def build_request_payload(
    *,
    runtime: Mapping[str, Any],
    system_prompt: str,
    api_content: list[dict[str, Any]],
    output_format: dict[str, Any],
) -> dict[str, Any]:
    """Compose one request from the centralized generation/provider settings."""
    generation = runtime.get("generation", {})
    provider = runtime.get("provider", {})
    if not isinstance(generation, Mapping):
        raise TypeError("runtime.generation must be an object")
    if not isinstance(provider, Mapping):
        raise TypeError("runtime.provider must be an object")
    reserved = {"model", "messages", "provider", "response_format"}
    if reserved & set(generation):
        raise ValueError("generation settings contain reserved request keys")

    payload: dict[str, Any] = {
        "model": runtime["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": api_content},
        ],
        **dict(generation),
        "response_format": output_format,
    }
    if provider:
        payload["provider"] = dict(provider)
    return payload


def call_model(
    *,
    runtime: Mapping[str, Any],
    api_key: str,
    system_prompt: str,
    api_content: list[dict[str, Any]],
    trace_content: list[dict[str, Any]],
    output_format: dict[str, Any],
    validator: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    """Execute one model call and return status, selection, and a lossless trace."""
    endpoint = endpoint_url(runtime["api_base"])
    payload = build_request_payload(
        runtime=runtime,
        system_prompt=system_prompt,
        api_content=api_content,
        output_format=output_format,
    )
    trace: dict[str, Any] = {
        "endpoint": endpoint,
        "request": {
            **{key: value for key, value in payload.items() if key != "messages"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": trace_content},
            ],
        },
        "response_status": None,
        "response_headers": None,
        "raw_response": None,
        "raw_response_text": None,
        "generations": [],
        "selected_generation_ordinal": None,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": runtime.get("user_agent", "vinishko-ndr-runner/1"),
    }
    http_referer_env = runtime.get("http_referer_env", "OPENROUTER_SITE_URL")
    app_title_env = runtime.get("app_title_env", "OPENROUTER_APP_TITLE")
    if site_url := os.environ.get(http_referer_env):
        headers["HTTP-Referer"] = site_url
    if app_title := os.environ.get(app_title_env):
        headers["X-Title"] = app_title
    try:
        http_request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(
            http_request,
            timeout=float(runtime.get("timeout", 120)),
        ) as response:
            raw_text = response.read().decode("utf-8", errors="replace")
            trace["response_status"] = response.status
            trace["response_headers"] = dict(response.headers.items())
            trace["raw_response_text"] = raw_text
        raw_response = json.loads(raw_text)
        trace["raw_response"] = raw_response
        records, selected_index, selected = generation_records(raw_response, validator)
        trace["generations"] = records
        trace["selected_generation_ordinal"] = selected_index
        if selected is None:
            return {
                "status": "contract_error",
                "error": "no generation satisfied this stage response contract",
                "selected": None,
                "trace": trace,
            }
        return {"status": "ok", "error": None, "selected": selected, "trace": trace}
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        trace["response_status"] = error.code
        trace["response_headers"] = dict(error.headers.items()) if error.headers else {}
        trace["raw_response_text"] = body
        try:
            trace["raw_response"] = json.loads(body)
        except json.JSONDecodeError:
            pass
        return {
            "status": "predictor_error",
            "error": f"HTTPError {error.code}: {error.reason}",
            "selected": None,
            "trace": trace,
        }
    except Exception as error:  # noqa: BLE001 - the caller persists this stage failure.
        return {
            "status": "predictor_error",
            "error": f"{type(error).__name__}: {error}",
            "selected": None,
            "trace": trace,
        }


def failure(status: str, error: str, trace: dict[str, Any]) -> dict[str, Any]:
    """Return a runner envelope for a failed pipeline stage."""
    return {
        "slug": None,
        "_status": status,
        "_error": error,
        "_trace": trace,
    }


def predict(request: dict[str, Any]) -> dict[str, Any]:
    """Compare every element, then resolve multiple matches when necessary."""
    runtime = request["runtime"]
    model = runtime.get("model") or os.environ.get("OPENROUTER_MODEL")
    api_key_env = runtime.get("api_key_env", "OPENROUTER_API_KEY")
    api_key = os.environ.get(api_key_env)
    runtime = {**runtime, "model": model}
    trace: dict[str, Any] = {
        "pipeline": "compare_each_then_resolve",
        "provider": "openrouter_compatible",
        "model": model,
        "comparison_calls": [],
        "same_slugs": [],
        "resolution_call": None,
    }
    if not model:
        return failure(
            "predictor_error",
            "model is required: pass --model or set OPENROUTER_MODEL",
            trace,
        )
    if not api_key:
        return failure(
            "predictor_error",
            f"environment variable {api_key_env} is not set",
            trace,
        )

    query_path = request["query"]["image_path"]
    image_detail = runtime.get("image_detail", "original")
    same_candidates = []
    for candidate in request["group"]["candidates"]:
        year_matters = bool(str(candidate.get("vintage", "")).strip())
        prompt_key = "compare_year_matters" if year_matters else "compare_year_not_matter"
        schema_key = "year_matters" if year_matters else "year_not_matter"
        api_content, trace_content = comparison_content(
            query_path,
            candidate,
            image_detail,
        )
        output_model = comparison_output_model(year_matters)
        result = call_model(
            runtime=runtime,
            api_key=api_key,
            system_prompt=request["prompts"][prompt_key]["content"],
            api_content=api_content,
            trace_content=trace_content,
            output_format=response_format(f"ndr_{schema_key}", output_model),
            validator=lambda payload, contract=output_model: validate_output(
                payload,
                contract,
            ),
        )
        trace["comparison_calls"].append(
            {
                "candidate_slug": candidate["slug"],
                "year_matters": year_matters,
                **result,
            },
        )
        if result["status"] != "ok":
            return failure(result["status"], result["error"], trace)
        if result["selected"]["verdict"] == "same":
            same_candidates.append(candidate)

    trace["same_slugs"] = [candidate["slug"] for candidate in same_candidates]
    if not same_candidates:
        return {
            "slug": NOT_FOUND,
            "_status": "ok",
            "_error": None,
            "_trace": trace,
        }
    if len(same_candidates) == 1:
        return {
            "slug": same_candidates[0]["slug"],
            "_status": "ok",
            "_error": None,
            "_trace": trace,
        }

    allowed_slugs = [candidate["slug"] for candidate in same_candidates]
    api_content, trace_content = resolution_content(
        query_path,
        same_candidates,
        image_detail,
    )
    resolve_model = resolver_output_model(allowed_slugs)
    resolution = call_model(
        runtime=runtime,
        api_key=api_key,
        system_prompt=request["prompts"]["resolve_multiple_same"]["content"],
        api_content=api_content,
        trace_content=trace_content,
        output_format=response_format("ndr_resolve_multiple_same", resolve_model),
        validator=lambda payload: validate_output(payload, resolve_model),
    )
    trace["resolution_call"] = resolution
    if resolution["status"] != "ok":
        return failure(resolution["status"], resolution["error"], trace)
    return {
        "slug": resolution["selected"]["slug"],
        "_status": "ok",
        "_error": None,
        "_trace": trace,
    }
