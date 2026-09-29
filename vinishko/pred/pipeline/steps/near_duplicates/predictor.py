"""One-call OpenRouter-compatible candidate-or-not-found predictor from NDR v5.

A call has a task: `group` selects within one near-duplicate group, `final` selects among the finalists of different groups
or verifies a single candidate. Both share the output contract and the vintage policy; they differ in the main prompt and the
closing instruction.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from typing import Literal

from dotenv import load_dotenv
from kostyl.utils import setup_logger
from tenacity import RetryCallState
from tenacity import Retrying
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import stop_before_delay
from tenacity import wait_exponential_jitter

from vinishko.openrouter_proxy import describe_error
from vinishko.openrouter_proxy import opener

from .models import ModelContractError
from .models import nearest_output_model
from .models import response_format
from .models import validate_output


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
Task = Literal["group", "final"]
"""Call task: select within one near-duplicate group, or among the finalists of different groups."""
TASK_GROUP: Task = "group"
TASK_FINAL: Task = "final"
TASK_PROMPTS = {
    TASK_GROUP: "resolve_multiple_same",
    TASK_FINAL: "resolve_finalists",
}
"""Main prompt file of every task."""
TASK_INSTRUCTIONS = {
    TASK_GROUP: "Compare the printed product words, then select exactly one nearest ELEMENT for QUERY from the complete group, "
    "or return not_found when every ELEMENT has a confident product-identity "
    "conflict. Return only JSON matching the structured output schema.",
    TASK_FINAL: "Compare the printed product words, then select the one ELEMENT that is the same product as QUERY, "
    "or return not_found when every ELEMENT has a confident product-identity "
    "conflict. Return only JSON matching the structured output schema.",
}
"""Closing instruction after the images of every task."""
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
"""HTTP answers repeated within the call deadline: rate limiting and transient provider failures."""
RETRY_WAIT_INITIAL_SECONDS, RETRY_WAIT_MAX_SECONDS = 0.5, 4.0
"""Backoff between attempts: 0.5, 1, 2, 4 s plus up to 1 s of jitter; a longer Retry-After from the server wins."""
RESPONSE_READ_CHUNK_BYTES = 64 * 1024
PROJECT_ROOT = Path(__file__).resolve().parents[5]
logger = setup_logger(fmt="detailed")


def load_openrouter_env() -> None:
    """Load project .env while keeping explicitly set environment variables."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def missing_api_key_message(api_key_env: str) -> str:
    """Explain where to set the missing OpenRouter key."""
    return f"{api_key_env} не задан: добавьте ключ в .env в корне проекта или задайте переменную окружения"


class RetryableHTTPError(Exception):
    """A throttled or transient answer worth another attempt within the call deadline."""

    def __init__(self, code: int, reason: str, retry_after: float | None) -> None:
        super().__init__(f"HTTPError {code}: {reason}")
        self.code = code
        self.retry_after = retry_after


def retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    """Seconds from a numeric Retry-After header; the HTTP-date form is ignored."""
    value = next(
        (v for k, v in headers.items() if k.lower() == "retry-after"), ""
    ).strip()
    return float(value) if value.replace(".", "", 1).isdigit() else None


class ModelCallTimeoutError(TimeoutError):
    """A strict wall-clock timeout with any response bytes read so far."""

    def __init__(self, timeout_seconds: float, partial_body: bytes = b"") -> None:
        super().__init__(
            f"model call exceeded {timeout_seconds:g}s wall-clock deadline",
        )
        self.partial_body = partial_body


def remaining_seconds(
    deadline: float,
    timeout_seconds: float,
    partial_body: bytes = b"",
) -> float:
    """Return time left before the model-call deadline or raise."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ModelCallTimeoutError(timeout_seconds, partial_body)
    return remaining


def set_response_read_timeout(response: Any, timeout_seconds: float) -> None:
    """Apply the shrinking deadline to urllib's underlying response socket."""
    stream = getattr(response, "fp", None)
    raw = getattr(stream, "raw", None)
    sock = getattr(raw, "_sock", None)
    setter = getattr(sock, "settimeout", None)
    if callable(setter):
        setter(timeout_seconds)


def read_response_bytes(
    response: Any,
    *,
    deadline: float,
    timeout_seconds: float,
) -> bytes:
    """Read a response while enforcing one wall-clock deadline across heartbeats."""
    chunks: list[bytes] = []
    read1 = getattr(response, "read1", None)
    if not callable(read1):
        remaining = remaining_seconds(deadline, timeout_seconds)
        set_response_read_timeout(response, remaining)
        try:
            body = response.read()
        except TimeoutError as error:
            raise ModelCallTimeoutError(timeout_seconds) from error
        if not isinstance(body, bytes):
            raise TypeError("HTTP response body must be bytes")
        remaining_seconds(deadline, timeout_seconds, body)
        return body

    while True:
        partial_body = b"".join(chunks)
        remaining = remaining_seconds(deadline, timeout_seconds, partial_body)
        set_response_read_timeout(response, remaining)
        try:
            chunk = read1(RESPONSE_READ_CHUNK_BYTES)
        except TimeoutError as error:
            raise ModelCallTimeoutError(timeout_seconds, partial_body) from error
        if not isinstance(chunk, bytes):
            # Test doubles and unusual wrappers may expose a synthetic read1.
            remaining = remaining_seconds(deadline, timeout_seconds, partial_body)
            set_response_read_timeout(response, remaining)
            try:
                body = response.read()
            except TimeoutError as error:
                raise ModelCallTimeoutError(timeout_seconds, partial_body) from error
            if not isinstance(body, bytes):
                raise TypeError("HTTP response body must be bytes")
            complete_body = partial_body + body
            remaining_seconds(deadline, timeout_seconds, complete_body)
            return complete_body
        if not chunk:
            return partial_body
        chunks.append(chunk)
        partial_body += chunk
        remaining_seconds(deadline, timeout_seconds, partial_body)


def provider_error_description(error: Any) -> str:
    """Describe an OpenRouter-compatible top-level error envelope."""
    if not isinstance(error, Mapping):
        return f"OpenRouter error: {error}"
    code = error.get("code")
    message = error.get("message") or "unknown provider error"
    if code is None:
        return f"OpenRouter error: {message}"
    return f"OpenRouter error {code}: {message}"


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
    source: str | bytes,
    detail: str = "original",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an API image block and a base64-free trace block."""
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    media_type = image_media_type(data)
    api_block = {
        "type": "image_url",
        "image_url": {
            "url": (
                f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"
            ),
            "detail": detail,
        },
    }
    trace_block = {
        "type": "image_url",
        "image": {
            "path": source if isinstance(source, str) else None,
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
    source: str | bytes,
    detail: str,
) -> None:
    """Append an encoded API image and a compact trace descriptor."""
    api_block, trace_block = image_blocks(source, detail)
    api_content.append(api_block)
    trace_content.append(trace_block)


def selection_content(
    query_image: str | bytes,
    candidates: list[Mapping[str, Any]],
    instruction: str,
    image_detail: str = "original",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the QUERY-versus-all-candidates multimodal message."""
    api_content: list[dict[str, Any]] = []
    trace_content: list[dict[str, Any]] = []
    add_text(api_content, trace_content, "=== QUERY ===")
    add_image(api_content, trace_content, query_image, image_detail)
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
            candidate.get(
                "reference_image_bytes", candidate.get("reference_image_path")
            ),
            image_detail,
        )
    add_text(api_content, trace_content, instruction)
    return api_content, trace_content


def selection_system_prompt(
    prompts: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    main_prompt: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Add the exact per-element vintage policy to the task's main prompt."""
    policies = []
    for number, candidate in enumerate(candidates, start=1):
        vintage = str(candidate.get("vintage", "")).strip()
        policies.append(
            {
                "element": number,
                "slug": candidate["slug"],
                "vintage": vintage,
                "exact_year_required": bool(vintage),
            },
        )

    prompt_parts = [prompts[main_prompt]["content"].rstrip()]
    if any(policy["exact_year_required"] for policy in policies):
        prompt_parts.append(prompts["compare_year_matters"]["content"].strip())
    if any(not policy["exact_year_required"] for policy in policies):
        prompt_parts.append(prompts["compare_year_not_matter"]["content"].strip())

    policy_lines = ["Vintage policy for every supplied element:"]
    for policy in policies:
        if policy["exact_year_required"]:
            rule = f"an exact match with {policy['vintage']} is required"
        else:
            rule = "an exact vintage match is not required"
        policy_lines.append(
            f"- ELEMENT {policy['element']} ({policy['slug']}): {rule}."
        )
    prompt_parts.append("\n".join(policy_lines))
    return "\n\n".join(prompt_parts), policies


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
    parsed = urllib.parse.urlparse(api_base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("api_base must be an HTTP(S) URL")
    normalized = api_base.rstrip("/")
    return (
        normalized
        if normalized.endswith("/chat/completions")
        else f"{normalized}/chat/completions"
    )


def system_prompt_with_schema(
    system_prompt: str,
    output_format: Mapping[str, Any],
) -> str:
    """Expose the exact response-format schema to models that cannot see it."""
    json_schema = output_format.get("json_schema")
    if not isinstance(json_schema, Mapping):
        raise TypeError("output_format.json_schema must be an object")
    schema = json_schema.get("schema")
    if not isinstance(schema, Mapping):
        raise TypeError("output_format.json_schema.schema must be an object")
    schema_name = json_schema.get("name", "structured_output")
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
    return (
        f"{system_prompt.rstrip()}\n\n"
        "The exact schema of the only valid JSON response is shown below. "
        "Follow every required field, enum, type, and additionalProperties rule; "
        "do not reproduce the schema itself.\n"
        f"SCHEMA_NAME: {schema_name}\n"
        f"JSON_SCHEMA:\n{schema_text}\n"
    )


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


def call_model(  # noqa: C901 - сохранён контракт и обработка ошибок NDR v5.
    *,
    runtime: Mapping[str, Any],
    api_key: str,
    system_prompt: str,
    api_content: list[dict[str, Any]],
    trace_content: list[dict[str, Any]],
    output_format: dict[str, Any],
    validator: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    """Execute one model call within a strict total deadline."""
    endpoint = endpoint_url(runtime["api_base"])
    effective_system_prompt = system_prompt_with_schema(system_prompt, output_format)
    payload = build_request_payload(
        runtime=runtime,
        system_prompt=effective_system_prompt,
        api_content=api_content,
        output_format=output_format,
    )
    trace: dict[str, Any] = {
        "endpoint": endpoint,
        "request": {
            **{key: value for key, value in payload.items() if key != "messages"},
            "messages": [
                {"role": "system", "content": effective_system_prompt},
                {"role": "user", "content": trace_content},
            ],
        },
        "response_status": None,
        "response_headers": None,
        "raw_response": None,
        "raw_response_text": None,
        "generations": [],
        "selected_generation_ordinal": None,
        "retry_attempts": [],
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

    timeout_seconds = float(runtime.get("timeout", 120))
    retries = int(runtime["retries"])
    deadline = time.monotonic() + timeout_seconds
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def attempt() -> dict[str, Any]:
        """One POST: the decoded response, or RetryableHTTPError for throttling and transient failures."""
        http_request = urllib.request.Request(  # noqa: S310 - endpoint_url проверяет схему.
            endpoint,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            # без OR_PROXY opener() — тот же, что у urllib.request.urlopen: прокси из окружения
            response_context = opener().open(
                http_request,
                timeout=remaining_seconds(deadline, timeout_seconds),
            )
            with response_context as response:
                trace["response_status"] = response.status
                trace["response_headers"] = dict(response.headers.items())
                raw_body = read_response_bytes(
                    response,
                    deadline=deadline,
                    timeout_seconds=timeout_seconds,
                )
        except urllib.error.HTTPError as error:
            trace["response_status"] = error.code
            trace["response_headers"] = (
                dict(error.headers.items()) if error.headers else {}
            )
            raw_text = read_response_bytes(
                error,
                deadline=deadline,
                timeout_seconds=timeout_seconds,
            ).decode("utf-8", errors="replace")
            trace["raw_response_text"] = raw_text
            try:
                trace["raw_response"] = json.loads(raw_text)
            except json.JSONDecodeError:
                trace["raw_response"] = None
            if error.code in RETRYABLE_STATUS:
                raise RetryableHTTPError(
                    error.code,
                    str(error.reason),
                    retry_after_seconds(trace["response_headers"]),
                ) from error
            raise
        raw_text = raw_body.decode("utf-8", errors="replace")
        trace["raw_response_text"] = raw_text
        raw_response = json.loads(raw_text)
        trace["raw_response"] = raw_response
        provider_error = raw_response.get("error")
        if (
            isinstance(provider_error, Mapping)
            and provider_error.get("code") in RETRYABLE_STATUS
        ):
            raise RetryableHTTPError(
                int(provider_error["code"]),
                str(provider_error.get("message", "provider error")),
                None,
            )
        return raw_response

    backoff = wait_exponential_jitter(
        initial=RETRY_WAIT_INITIAL_SECONDS, max=RETRY_WAIT_MAX_SECONDS
    )

    def wait(state: RetryCallState) -> float:
        error = state.outcome.exception() if state.outcome is not None else None
        hinted = error.retry_after if isinstance(error, RetryableHTTPError) else None
        return max(backoff(state), hinted or 0.0)

    def before_sleep(state: RetryCallState) -> None:
        error = state.outcome.exception() if state.outcome is not None else None
        code = error.code if isinstance(error, RetryableHTTPError) else None
        trace["retry_attempts"].append(
            {
                "attempt": state.attempt_number,
                "response_status": code,
                "response_headers": trace["response_headers"],
                "raw_response": trace["raw_response"],
                "raw_response_text": trace["raw_response_text"],
                "delay_seconds": state.upcoming_sleep,
            },
        )
        logger.warning(
            f"OpenRouter ответил {code}: повтор через {state.upcoming_sleep:.1f} с, попытка {state.attempt_number + 1} из {retries + 1}"
        )

    retrying = Retrying(
        retry=retry_if_exception_type(RetryableHTTPError),
        wait=wait,
        stop=stop_after_attempt(retries + 1) | stop_before_delay(timeout_seconds),
        before_sleep=before_sleep,
        reraise=True,
    )
    try:
        raw_response = retrying(attempt)
        provider_error = raw_response.get("error")
        if provider_error is not None:
            trace["provider_error"] = provider_error
            return {
                "status": "predictor_error",
                "error": provider_error_description(provider_error),
                "selected": None,
                "trace": trace,
            }

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
    except RetryableHTTPError as error:
        return {
            "status": "predictor_error",
            "error": f"{error}; попыток {len(trace['retry_attempts']) + 1}",
            "selected": None,
            "trace": trace,
        }
    except urllib.error.HTTPError as error:
        return {
            "status": "predictor_error",
            "error": f"HTTPError {error.code}: {error.reason}",
            "selected": None,
            "trace": trace,
        }
    except ModelCallTimeoutError as error:
        if error.partial_body:
            trace["raw_response_text"] = error.partial_body.decode(
                "utf-8",
                errors="replace",
            )
        trace["timeout"] = {
            "kind": "wall_clock",
            "limit_seconds": timeout_seconds,
        }
        return {
            "status": "predictor_error",
            "error": f"TimeoutError: {error}",
            "selected": None,
            "trace": trace,
        }
    except TimeoutError as error:
        trace["timeout"] = {
            "kind": "wall_clock",
            "limit_seconds": timeout_seconds,
        }
        return {
            "status": "predictor_error",
            "error": f"TimeoutError: {error}",
            "selected": None,
            "trace": trace,
        }
    except Exception as error:
        return {
            "status": "predictor_error",
            "error": describe_error(error),
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
    """Choose one of the request candidates or not_found in one call of the request task."""
    load_openrouter_env()
    task = request["task"]
    runtime = request["runtime"]
    model = runtime.get("model") or os.environ.get("OPENROUTER_MODEL")
    api_key_env = runtime.get("api_key_env", "OPENROUTER_API_KEY")
    api_key = (os.environ.get(api_key_env) or "").strip()
    runtime = {**runtime, "model": model}
    trace: dict[str, Any] = {
        "pipeline": "select_nearest_once",
        "task": task,
        "provider": "openrouter_compatible",
        "model": model,
        "comparison_calls": [],
        "candidate_slugs": [],
        "year_policies": [],
    }
    if task not in TASK_PROMPTS:
        return failure("predictor_error", f"unknown task {task!r}", trace)
    if not model:
        return failure(
            "predictor_error",
            "model is required: pass --model or set OPENROUTER_MODEL",
            trace,
        )
    if not api_key:
        return failure(
            "predictor_error",
            missing_api_key_message(api_key_env),
            trace,
        )

    query = request["query"]
    query_image: str | bytes = (
        query["image_bytes"] if "image_bytes" in query else query["image_path"]
    )
    image_detail: str = runtime.get("image_detail") or "original"
    candidates = list(request["candidates"])
    allowed_slugs = [candidate["slug"] for candidate in candidates]
    trace["candidate_slugs"] = allowed_slugs
    try:
        output_model = nearest_output_model(allowed_slugs)
    except ModelContractError as error:
        return failure("predictor_error", str(error), trace)

    system_prompt, year_policies = selection_system_prompt(
        request["prompts"],
        candidates,
        TASK_PROMPTS[task],
    )
    trace["year_policies"] = year_policies
    api_content, trace_content = selection_content(
        query_image,
        candidates,
        TASK_INSTRUCTIONS[task],
        image_detail,
    )
    result = call_model(
        runtime=runtime,
        api_key=api_key,
        system_prompt=system_prompt,
        api_content=api_content,
        trace_content=trace_content,
        output_format=response_format(f"ndr_select_{task}", output_model),
        validator=lambda payload: validate_output(payload, output_model),
    )
    trace["comparison_calls"].append(
        {
            "stage": task,
            "candidate_slugs": allowed_slugs,
            **result,
        },
    )
    if result["status"] != "ok":
        return failure(result["status"], result["error"], trace)
    return {
        "slug": result["selected"]["slug"],
        "_status": "ok",
        "_error": None,
        "_trace": trace,
    }
