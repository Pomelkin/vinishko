"""Exercise a running sommelier service over HTTP."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
import uuid

import httpx


def new_uuid7() -> uuid.UUID:
    """Generate a UUIDv7 on Python 3.13, which has no uuid.uuid7 yet."""
    timestamp_ms = time.time_ns() // 1_000_000
    random_bits = secrets.randbits(74)
    value = (timestamp_ms & ((1 << 48) - 1)) << 80
    value |= 7 << 76
    value |= ((random_bits >> 62) & 0xFFF) << 64
    value |= 0b10 << 62
    value |= random_bits & ((1 << 62) - 1)
    return uuid.UUID(int=value)


def send(client: httpx.Client, method: str, path: str, **kwargs: object) -> dict | None:
    """Print one HTTP exchange and fail with the response body on errors."""
    response = client.request(method, path, **kwargs)
    print(f"{method} {path} -> {response.status_code}")
    if response.is_error:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    return response.json() if response.content else None


def main() -> None:
    if os.name == "nt":
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Проверка работающего API сомелье по HTTP")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SOMMELIER_BASE_URL", "http://127.0.0.1:8000"),
        help="Адрес сервиса (по умолчанию http://127.0.0.1:8000)",
    )
    parser.add_argument("--question", default="С чем лучше сочетать это вино?")
    parser.add_argument("--keep", action="store_true", help="Оставить созданную сессию после проверки")
    args = parser.parse_args()

    session_id = str(new_uuid7())
    path = f"/v1/sessions/{session_id}"
    wine = {
        "Название вина": "100 оттенков красного. Каберне",
        "Категория": "Красное",
        "Цвет": "Темно-рубиновый",
        "Регион": "Кубань",
        "Сорт винограда": "Каберне Совиньон",
        "Описание": "В аромате спелая вишня, чёрная смородина, лёгкая пряность. Вкус плотный, гармоничный, с мягкими танинами и ягодным послевкусием.",
        "Винодельня": "Фанагория",
        "Slug": "fanagoriya-100-ottenkov-krasnogo-kaberne-kaberne-sovinon-krasnoe-suhoe-135",
        "Крепость, %": 13.5,
        "Сахар": "сухое",
    }
    candidates = [
        {
            "Название вина": "100 оттенков красного. Пино Нуар",
            "Категория": "Красное",
            "Регион": "Кубань",
            "Сорт винограда": "Пино Нуар",
            "Описание": "Стройный развитый аромат с нотами брусники, вяленых лесных ягод, гранатовых зерен, пастилы, тонкими животными нюансами. Вкус легкий и волнующий, многослойный. Все элементы вина находятся в балансе. Тонко, свежо, элегантно.",
            "Винодельня": "Фанагория",
            "Slug": "fanagoriya-100-ottenkov-krasnogo-pino-nuar-krasnoe-suhoe-135",
            "Крепость, %": 13.5,
            "Сахар": "сухое",
        },
        {
            "Название вина": "100 оттенков красного. Саперави",
            "Категория": "Красное",
            "Регион": "Кубань",
            "Сорт винограда": "Саперави",
            "Описание": "В аромате тона ежевики, черники, тёмной вишни, оттенки специй. Вкус насыщенный, сбалансированный, с бархатными танинами.",
            "Винодельня": "Фанагория",
            "Slug": "fanagoriya-100-ottenkov-krasnogo-saperavi-krasnoe-suhoe-14",
            "Крепость, %": 14,
            "Сахар": "сухое",
        },
    ]
    opened = False

    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=90.0, trust_env=False) as client:
        health = send(client, "GET", "/health")
        assert health == {"status": "ok"}, health

        try:
            opening = send(client, "POST", path, json={"wine": wine, "candidates": candidates})
            opened = True
            assert opening["session_id"] == session_id
            assert opening["message"]["role"] == "assistant"
            assert len(opening["message"]["suggestions"]) == 2
            print(json.dumps(opening, ensure_ascii=False, indent=2))

            reply = send(client, "POST", f"{path}/messages", json={"content": args.question})
            assert reply["session_id"] == session_id
            assert reply["message"]["role"] == "assistant"
            assert reply["message"]["suggestions"] is None
            print(json.dumps(reply, ensure_ascii=False, indent=2))

            history = send(client, "GET", path)
            assert [message["role"] for message in history["messages"]] == [
                "assistant", "user", "assistant",
            ]
            assert history["messages"][1]["content"] == args.question
            print(json.dumps(history, ensure_ascii=False, indent=2))
            print(f"Проверка прошла. session_id={session_id}")
        finally:
            if opened and not args.keep:
                send(client, "DELETE", path)


if __name__ == "__main__":
    main()
