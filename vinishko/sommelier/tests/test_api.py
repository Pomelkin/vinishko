"""Integration checks for the HTTP contract and persisted provider history."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from fastapi import FastAPI
from httpx import ASGITransport
from httpx import AsyncClient

from vinishko.sommelier.router import router
from vinishko.sommelier.service import SommelierService
from vinishko.sommelier.storage import JsonSessionStore


SESSION_ID = "01993959-0000-7000-8000-000000000001"
OTHER_SESSION_ID = "01993959-0000-7000-8000-000000000002"
WINE = {"Название вина": "Кокур", "Slug": "kokur", "Крепость, %": "12"}
CANDIDATE = {"Название вина": "Другой кокур", "Slug": "other-kokur"}
FIRST_CONTENT = "Кокур — белое вино.\nЧем я могу вам помочь?"


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.fail = False
        self.first_suggestions = ["Как подавать", "Какой вкус"]

    def __call__(self, request: dict) -> dict:
        self.calls.append(request)
        if self.fail:
            return {"_status": "provider_error", "_error": "upstream unavailable"}
        if not request["history"]:
            return {
                "_status": "ok",
                "content": FIRST_CONTENT,
                "suggestions": self.first_suggestions,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "content": FIRST_CONTENT,
                            "suggestions": self.first_suggestions,
                        },
                        ensure_ascii=False,
                    ),
                    "reasoning": "internal provider context",
                    "reasoning_details": [
                        {"type": "reasoning.text", "text": "raw details"}
                    ],
                },
            }
        content = f"Ответ {len(self.calls) - 1}"
        return {
            "_status": "ok",
            "content": content,
            "suggestions": None,
            "message": {"role": "assistant", "content": content},
        }


def create_app(service: SommelierService) -> FastAPI:
    """Only the sommelier routes, without the recognition pipeline of the main application."""
    app = FastAPI()
    app.include_router(router)
    app.state.sommelier = service
    return app


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.engine = FakeEngine()
        self.service = SommelierService(JsonSessionStore(self.directory), self.engine)
        self.client = AsyncClient(
            transport=ASGITransport(app=create_app(self.service)),
            base_url="http://test",
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        self.temp.cleanup()

    async def open(self):
        return await self.client.post(
            f"/sommelier/sessions/{SESSION_ID}",
            json={"wine": WINE, "candidates": [CANDIDATE]},
        )

    async def test_open_follow_up_and_recover_from_disk(self) -> None:
        opening = await self.open()
        self.assertEqual(opening.status_code, 201, opening.text)
        self.assertEqual(
            opening.json()["message"],
            {
                "role": "assistant",
                "content": FIRST_CONTENT,
                "suggestions": ["Как подавать", "Какой вкус"],
            },
        )
        self.assertNotIn("reasoning", opening.text)

        # Simulate a process restart: a fresh service reads only the JSON file.
        new_service = SommelierService(JsonSessionStore(self.directory), self.engine)
        async with AsyncClient(
            transport=ASGITransport(app=create_app(new_service)), base_url="http://test"
        ) as client:
            reply = await client.post(
                f"/sommelier/sessions/{SESSION_ID}/messages",
                json={"content": "  С чем подавать?  "},
            )
            self.assertEqual(reply.status_code, 201, reply.text)
            self.assertEqual(reply.json()["message"]["content"], "Ответ 1")
            session = await client.get(f"/sommelier/sessions/{SESSION_ID}")
            self.assertEqual(session.status_code, 200)
            self.assertEqual(
                [m["role"] for m in session.json()["messages"]],
                ["assistant", "user", "assistant"],
            )
            self.assertEqual(
                session.json()["messages"][1]["content"], "С чем подавать?"
            )
            self.assertNotIn("reasoning", session.text)

        history = self.engine.calls[1]["history"]
        self.assertEqual(history[0]["reasoning"], "internal provider context")
        self.assertEqual(history[0]["reasoning_details"][0]["text"], "raw details")
        self.assertEqual(history[1], {"role": "user", "content": "С чем подавать?"})
        self.assertEqual(self.engine.calls[1]["wine"], WINE)
        self.assertEqual(self.engine.calls[1]["candidates"], [CANDIDATE])
        stored = json.loads(
            (self.directory / f"{SESSION_ID}.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            stored["messages"][0]["provider_message"]["reasoning"],
            "internal provider context",
        )

    async def test_validation_conflict_and_delete(self) -> None:
        self.assertEqual(
            (await self.client.get(f"/sommelier/sessions/{SESSION_ID}")).status_code,
            404,
        )
        self.assertEqual(
            (
                await self.client.post(
                    f"/sommelier/sessions/{SESSION_ID}/messages",
                    json={"content": "Привет"},
                )
            ).status_code,
            404,
        )
        self.assertEqual(
            (
                await self.client.post(
                    "/sommelier/sessions/01993959-0000-4000-8000-000000000001",
                    json={"wine": WINE, "candidates": []},
                )
            ).status_code,
            422,
        )
        self.assertEqual(
            (
                await self.client.post(
                    f"/sommelier/sessions/{SESSION_ID}",
                    json={"wine": WINE, "candidates": [CANDIDATE] * 6},
                )
            ).status_code,
            422,
        )
        self.assertEqual((await self.open()).status_code, 201)
        self.assertEqual((await self.open()).status_code, 409)
        self.assertEqual(
            (
                await self.client.post(
                    f"/sommelier/sessions/{SESSION_ID}/messages",
                    json={"content": "  "},
                )
            ).status_code,
            422,
        )
        self.assertEqual(
            (await self.client.delete(f"/sommelier/sessions/{SESSION_ID}")).status_code,
            204,
        )
        self.assertEqual(
            (await self.client.get(f"/sommelier/sessions/{SESSION_ID}")).status_code,
            404,
        )
        self.assertEqual(
            (await self.client.delete(f"/sommelier/sessions/{SESSION_ID}")).status_code,
            404,
        )

    async def test_open_accepts_one_suggestion(self) -> None:
        self.engine.first_suggestions = ["Как подавать?"]
        opening = await self.open()
        self.assertEqual(opening.status_code, 201, opening.text)
        self.assertEqual(opening.json()["message"]["suggestions"], ["Как подавать?"])

    async def test_provider_failure_does_not_modify_session(self) -> None:
        self.engine.fail = True
        failed_open = await self.open()
        self.assertEqual(failed_open.status_code, 502)
        self.assertFalse((self.directory / f"{SESSION_ID}.json").exists())
        self.engine.fail = False
        self.assertEqual((await self.open()).status_code, 201)
        self.engine.fail = True
        failed_reply = await self.client.post(
            f"/sommelier/sessions/{SESSION_ID}/messages",
            json={"content": "Вопрос"},
        )
        self.assertEqual(failed_reply.status_code, 502)
        history = await self.client.get(f"/sommelier/sessions/{SESSION_ID}")
        self.assertEqual(len(history.json()["messages"]), 1)

    async def test_missing_provider_key_returns_503(self) -> None:
        service = SommelierService(JsonSessionStore(self.directory))
        async with AsyncClient(
            transport=ASGITransport(app=create_app(service)), base_url="http://test"
        ) as client:
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
                response = await client.post(
                    f"/sommelier/sessions/{SESSION_ID}",
                    json={"wine": WINE, "candidates": []},
                )
        self.assertEqual(response.status_code, 503, response.text)
        self.assertFalse((self.directory / f"{SESSION_ID}.json").exists())

    async def test_unreadable_secret_file_returns_503(self) -> None:
        service = SommelierService(JsonSessionStore(self.directory))
        async with AsyncClient(
            transport=ASGITransport(app=create_app(service)), base_url="http://test"
        ) as client:
            with patch.dict(
                os.environ,
                {"OPENROUTER_API_KEY_FILE": str(self.directory / "missing-key")},
            ):
                response = await client.post(
                    f"/sommelier/sessions/{SESSION_ID}",
                    json={"wine": WINE, "candidates": []},
                )
        self.assertEqual(response.status_code, 503, response.text)
        self.assertFalse((self.directory / f"{SESSION_ID}.json").exists())

    async def test_concurrent_follow_ups_are_serialized(self) -> None:
        self.assertEqual((await self.open()).status_code, 201)
        responses = await asyncio.gather(
            *[
                self.client.post(
                    f"/sommelier/sessions/{SESSION_ID}/messages",
                    json={"content": question},
                )
                for question in ("Первый вопрос", "Второй вопрос")
            ]
        )
        self.assertEqual([response.status_code for response in responses], [201, 201])
        self.assertEqual(len(self.engine.calls[1]["history"]), 2)
        self.assertEqual(len(self.engine.calls[2]["history"]), 4)
        stored = await self.service.store.load(UUID(SESSION_ID))
        self.assertEqual(len(stored.messages), 5)  # ty: ignore[unresolved-attribute]

    async def test_provider_wait_does_not_block_other_requests(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def blocking_engine(request: dict) -> dict:
            started.set()
            release.wait(timeout=5)
            return self.engine(request)

        service = SommelierService(JsonSessionStore(self.directory), blocking_engine)
        async with AsyncClient(
            transport=ASGITransport(app=create_app(service)), base_url="http://test"
        ) as client:
            opening = asyncio.create_task(
                client.post(
                    f"/sommelier/sessions/{SESSION_ID}",
                    json={"wine": WINE, "candidates": []},
                )
            )
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                self.assertFalse(opening.done())
                other = await asyncio.wait_for(
                    client.get(f"/sommelier/sessions/{OTHER_SESSION_ID}"), timeout=1
                )
                self.assertEqual(other.status_code, 404)
            finally:
                release.set()
                await opening


if __name__ == "__main__":
    unittest.main()
