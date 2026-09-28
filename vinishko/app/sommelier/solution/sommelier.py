"""Stateless DeepSeek/OpenRouter interface for the digital sommelier."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vinishko.app.sommelier.solution.config import SETTINGS
from vinishko.app.sommelier.solution.config import SommelierSettings
from vinishko.app.sommelier.solution.config import settings_with_overrides
from vinishko.app.sommelier.solution.knowledge import NO_DATA
from vinishko.app.sommelier.solution.knowledge import compile_expert_knowledge
from vinishko.app.sommelier.solution.knowledge import load_expert_knowledge
from vinishko.app.sommelier.solution.knowledge import normalize_catalog_card
from vinishko.app.sommelier.solution.models import OutputContractError
from vinishko.app.sommelier.solution.models import response_format
from vinishko.app.sommelier.solution.models import validate_first_turn


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
FIRST_TURN_TRIGGER = "Сформируй первое сообщение цифрового сомелье."
HISTORY_MESSAGE_FIELDS = (
    "role",
    "content",
    "reasoning",
    "reasoning_details",
    "refusal",
    "annotations",
)


@dataclass(frozen=True)
class PreparedRequest:
    """A provider payload plus local data required to validate its response."""

    payload: dict[str, Any]
    first_turn: bool
    settings: SommelierSettings
    selected_knowledge_values: dict[str, str]
    injected_expert_blocks: list[dict[str, Any]]
    prompt_sha256: str


def _prompt(name: str) -> str:
    """Read one version-controlled UTF-8 prompt fragment."""
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def _history(value: Any) -> list[dict[str, Any]]:
    """Validate and preserve OpenRouter reasoning fields across chat turns."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("request.history must be an array")

    history: list[dict[str, Any]] = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise TypeError(f"request.history[{index}] must be an object")
        role = message.get("role")
        if role not in {"user", "assistant"}:
            raise ValueError(
                f"request.history[{index}].role must be user or assistant",
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"request.history[{index}].content must be a non-empty string",
            )
        history.append(
            {
                field: message[field]
                for field in HISTORY_MESSAGE_FIELDS
                if field in message and message[field] is not None
            },
        )
    if history and history[-1]["role"] != "user":
        raise ValueError("a non-empty history must end with the current user message")
    return history


def _candidates(value: Any) -> list[Mapping[str, Any]]:
    """Validate the bounded candidate-card collection."""
    if not isinstance(value, list):
        raise TypeError("request.candidates must be an array")
    if len(value) > 5:
        raise ValueError("request.candidates must contain at most five cards")
    if any(not isinstance(card, Mapping) for card in value):
        raise TypeError("every candidate must be a catalog-card object")
    return value


def _context_text(
    wine: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    expert_blocks: list[dict[str, Any]],
    *,
    include_candidates: bool,
) -> str:
    """Serialize grounded context, with unknown fields made explicit."""
    expert_value: list[dict[str, Any]] | str = expert_blocks or NO_DATA
    sections = [
        "НАЙДЕННОЕ ВИНО (единственная основная карточка):\n"
        + json.dumps(normalize_catalog_card(wine), ensure_ascii=False, indent=2),
    ]
    if include_candidates:
        normalized_candidates = [
            {"rank": rank, "card": normalize_catalog_card(candidate)}
            for rank, candidate in enumerate(candidates, start=1)
        ]
        sections.append(
            "TOP-5 КАНДИДАТОВ (только возможные альтернативы):\n"
            + json.dumps(normalized_candidates, ensure_ascii=False, indent=2),
        )
    sections.append(
        "ЭКСПЕРТНЫЕ БЛОКИ (подобраны только по найденному вину):\n"
        + json.dumps(expert_value, ensure_ascii=False, indent=2),
    )
    return "\n\n".join(sections)


def _first_prompt_with_schema(prompt: str, output_format: Mapping[str, Any]) -> str:
    """Append the exact provider schema as the final system-prompt section."""
    schema_container = output_format.get("json_schema")
    if not isinstance(schema_container, Mapping):
        raise TypeError("response_format.json_schema must be an object")
    schema = schema_container.get("schema")
    if not isinstance(schema, Mapping):
        raise TypeError("response_format.json_schema.schema must be an object")
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
    return (
        f"{prompt.rstrip()}\n\n"
        "ТОЧНАЯ СХЕМА ЕДИНСТВЕННОГО ДОПУСТИМОГО ОТВЕТА:\n"
        f"SCHEMA_NAME: {schema_container['name']}\n"
        f"JSON_SCHEMA:\n{schema_text}"
    )


def prepare_request(
    request: Mapping[str, Any],
    *,
    base_settings: SommelierSettings = SETTINGS,
) -> PreparedRequest:
    """Build a complete OpenRouter payload without performing network I/O."""
    if not isinstance(request, Mapping):
        raise TypeError("request must be an object")
    wine = request.get("wine")
    if not isinstance(wine, Mapping):
        raise TypeError("request.wine must be a catalog-card object")
    candidates = _candidates(request.get("candidates"))
    history = _history(request.get("history", []))
    settings = settings_with_overrides(request.get("runtime"), base_settings)

    knowledge = load_expert_knowledge(settings.expert_knowledge_path)
    selections, expert_blocks = compile_expert_knowledge(wine, knowledge)
    first_turn = not history
    context = _context_text(
        wine,
        candidates,
        expert_blocks,
        include_candidates=not first_turn,
    )
    mode_prompt = _prompt("first_turn.txt" if first_turn else "follow_up.txt")
    system_prompt = "\n\n".join((_prompt("common.txt"), mode_prompt, context))

    output_format = response_format() if first_turn else None
    if output_format is not None:
        system_prompt = _first_prompt_with_schema(system_prompt, output_format)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]
    if first_turn:
        messages.append({"role": "user", "content": FIRST_TURN_TRIGGER})
    else:
        messages.extend(history)

    payload: dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
        **settings.generation.request_payload(),
        "provider": settings.provider.request_payload(),
    }
    if output_format is not None:
        payload["response_format"] = output_format

    return PreparedRequest(
        payload=payload,
        first_turn=first_turn,
        settings=settings,
        selected_knowledge_values=selections,
        injected_expert_blocks=expert_blocks,
        prompt_sha256=hashlib.sha256(system_prompt.encode()).hexdigest(),
    )


def _endpoint(api_base: str) -> str:
    """Normalize an API base into the Chat Completions endpoint."""
    normalized = api_base.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _headers(settings: SommelierSettings, api_key: str) -> dict[str, str]:
    """Build request headers without exposing the API key to traces."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": settings.user_agent,
    }
    if referer := os.environ.get(settings.http_referer_env):
        headers["HTTP-Referer"] = referer
    if title := os.environ.get(settings.app_title_env):
        headers["X-Title"] = title
    return headers


class ProviderKeyError(ValueError):
    """The OpenRouter key is missing or its secret file cannot be read."""


def _api_key(settings: SommelierSettings) -> str:
    """Read a mounted secret file when configured, otherwise use the local env var."""
    file_env = f"{settings.api_key_env}_FILE"
    if file_env in os.environ:
        secret_path = os.environ[file_env]
        if not secret_path:
            raise ProviderKeyError(f"{file_env} is empty")
        try:
            key = Path(secret_path).read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeError) as error:
            raise ProviderKeyError(f"cannot read {file_env}: {type(error).__name__}") from error
        if not key:
            raise ProviderKeyError(f"secret file from {file_env} is empty")
        return key

    key = os.environ.get(settings.api_key_env, "").strip()
    if not key:
        raise ProviderKeyError(f"environment variable {settings.api_key_env} is not set")
    return key


def _assistant_message(raw_response: Mapping[str, Any]) -> dict[str, Any]:
    """Return the first raw assistant message, retaining reasoning metadata."""
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OutputContractError("OpenRouter response has no choices")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise OutputContractError("OpenRouter choice must be an object")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise OutputContractError("OpenRouter choice has no message object")
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise OutputContractError("OpenRouter message has no textual content")
    return {"role": "assistant", **dict(message)}


def _failure(
    status: str,
    error: str,
    trace: dict[str, Any],
) -> dict[str, Any]:
    """Return one stable runner-friendly failure envelope."""
    return {
        "content": None,
        "suggestions": None,
        "message": None,
        "_status": status,
        "_error": error,
        "_trace": trace,
    }


def respond(request: Mapping[str, Any]) -> dict[str, Any]:  # noqa: C901
    """Call DeepSeek through OpenRouter and return a runner-friendly result."""
    try:
        prepared = prepare_request(request)
    except Exception as error:
        return _failure(
            "request_error",
            f"{type(error).__name__}: {error}",
            {"pipeline": "ai_sommelier"},
        )

    endpoint = _endpoint(prepared.settings.api_base)
    trace: dict[str, Any] = {
        "pipeline": "ai_sommelier",
        "prompt_sha256": prepared.prompt_sha256,
        "model": prepared.settings.model,
        "first_turn": prepared.first_turn,
        "selected_knowledge_values": prepared.selected_knowledge_values,
        "injected_expert_blocks": prepared.injected_expert_blocks,
        "endpoint": endpoint,
        "request": prepared.payload,
        "response_status": None,
        "raw_response": None,
        "attempts": [],
    }
    try:
        api_key = _api_key(prepared.settings)
    except ProviderKeyError as error:
        return _failure(
            "configuration_error",
            str(error),
            trace,
        )

    http_request = urllib.request.Request(  # noqa: S310 - configured endpoint.
        endpoint,
        data=json.dumps(prepared.payload, ensure_ascii=False).encode("utf-8"),
        headers=_headers(prepared.settings, api_key),
        method="POST",
    )
    max_attempts = 2 if prepared.first_turn else 1
    try:
        proxy = os.environ.get("openrouter_http_proxy", "").strip()
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy} if proxy else {})
        )
        for attempt_number in range(1, max_attempts + 1):
            attempt_trace: dict[str, Any] = {
                "number": attempt_number,
                "response_status": None,
                "raw_response": None,
            }
            trace["attempts"].append(attempt_trace)
            with opener.open(  # noqa: S310 - configured OpenRouter endpoint.
                http_request,
                timeout=prepared.settings.timeout_seconds,
            ) as response:
                trace["response_status"] = response.status
                attempt_trace["response_status"] = response.status
                raw_text = response.read().decode("utf-8", errors="replace")
            raw_response = json.loads(raw_text)
            trace["raw_response"] = raw_response
            attempt_trace["raw_response"] = raw_response
            try:
                if not isinstance(raw_response, Mapping):
                    raise OutputContractError("OpenRouter response must be an object")
                if raw_response.get("error") is not None:
                    return _failure(
                        "provider_error",
                        f"OpenRouter error: {raw_response['error']}",
                        trace,
                    )
                message = _assistant_message(raw_response)
                if prepared.first_turn:
                    try:
                        output = validate_first_turn(json.loads(message["content"]))
                    except json.JSONDecodeError as error:
                        raise OutputContractError(
                            f"invalid first-turn JSON: {error}",
                        ) from error
                    content = output["content"]
                    suggestions = output["suggestions"]
                else:
                    content = message["content"]
                    suggestions = None
            except OutputContractError as error:
                attempt_trace["validation_error"] = str(error)
                if prepared.first_turn and attempt_number == 1:
                    continue
                return _failure("contract_error", str(error), trace)
            return {
                "content": content,
                "suggestions": suggestions,
                "message": message,
                "_status": "ok",
                "_error": None,
                "_trace": trace,
            }
    except urllib.error.HTTPError as error:
        trace["response_status"] = error.code
        trace["attempts"][-1]["response_status"] = error.code
        raw_text = error.read().decode("utf-8", errors="replace")
        try:
            trace["raw_response"] = json.loads(raw_text)
            trace["attempts"][-1]["raw_response"] = trace["raw_response"]
        except json.JSONDecodeError:
            trace["raw_response_text"] = raw_text
            trace["attempts"][-1]["raw_response_text"] = raw_text
        return _failure(
            "provider_error",
            f"HTTPError {error.code}: {error.reason}",
            trace,
        )
    except (OSError, TimeoutError, json.JSONDecodeError) as error:
        trace["attempts"][-1]["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, json.JSONDecodeError):
            trace["raw_response_text"] = raw_text
            trace["attempts"][-1]["raw_response_text"] = raw_text
        return _failure(
            "provider_error",
            f"{type(error).__name__}: {error}",
            trace,
        )
