"""Session orchestration around the stateless v2 sommelier."""

from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException
from pydantic import UUID7, ValidationError

from app.solution.v2 import respond
from app.models import (
    OpenSessionRequest,
    SessionRecord,
    StoredMessage,
    TurnResponse,
    UserMessageRequest,
)
from app.storage import JsonSessionStore


logger = logging.getLogger(__name__)
SommelierEngine = Callable[[dict[str, Any]], dict[str, Any]]


class SommelierService:
    def __init__(self, store: JsonSessionStore, engine: SommelierEngine = respond) -> None:
        self.store = store
        self.engine = engine
        # A held lock keeps itself alive; idle session locks are garbage collected.
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def _lock(self, session_id: UUID7) -> asyncio.Lock:
        key = str(session_id)
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    async def _generate(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await asyncio.to_thread(self.engine, request)
        except Exception:
            logger.exception("Sommelier engine crashed")
            raise HTTPException(status_code=502, detail="Sommelier is temporarily unavailable") from None

        if not isinstance(result, dict):
            raise HTTPException(status_code=502, detail="Sommelier returned an invalid response")
        if result.get("_status") != "ok":
            status = result.get("_status")
            logger.error("Sommelier generation failed: %s: %s", status, result.get("_error"))
            if status == "request_error" and "OPENROUTER_API_KEY is not set" in str(result.get("_error")):
                raise HTTPException(status_code=503, detail="Sommelier provider is not configured")
            raise HTTPException(status_code=502, detail="Sommelier is temporarily unavailable")
        return result

    @staticmethod
    def _assistant(result: dict[str, Any]) -> StoredMessage:
        raw = result.get("message")
        content = result.get("content")
        suggestions = result.get("suggestions")
        if not isinstance(raw, dict) or not isinstance(content, str) or not content.strip():
            raise HTTPException(status_code=502, detail="Sommelier returned an invalid response")
        if raw.get("role") != "assistant":
            raise HTTPException(status_code=502, detail="Sommelier returned an invalid response")
        try:
            return StoredMessage(
                role="assistant",
                content=content,
                suggestions=suggestions,
                provider_message=raw,
            )
        except ValidationError:
            raise HTTPException(status_code=502, detail="Sommelier returned an invalid response") from None

    async def open(self, session_id: UUID7, request: OpenSessionRequest) -> TurnResponse:
        async with self._lock(session_id):
            if await self.store.load(session_id) is not None:
                raise HTTPException(status_code=409, detail="Session already exists")
            result = await self._generate({
                "wine": request.wine,
                "candidates": request.candidates,
                "history": [],
            })
            assistant = self._assistant(result)
            if assistant.suggestions is None or len(assistant.suggestions) != 2:
                raise HTTPException(status_code=502, detail="Sommelier returned an invalid response")
            record = SessionRecord(
                session_id=session_id,
                wine=request.wine,
                candidates=request.candidates,
                messages=[assistant],
            )
            await self.store.save(record)
            return TurnResponse(session_id=session_id, message=assistant.public())

    async def ask(self, session_id: UUID7, request: UserMessageRequest) -> TurnResponse:
        async with self._lock(session_id):
            record = await self.store.load(session_id)
            if record is None:
                raise HTTPException(status_code=404, detail="Session not found")
            result = await self._generate({
                "wine": record.wine,
                "candidates": record.candidates,
                "history": record.provider_history(request.content),
            })
            assistant = self._assistant(result)
            if assistant.suggestions is not None:
                raise HTTPException(status_code=502, detail="Sommelier returned an invalid response")
            record.messages.extend((
                StoredMessage(role="user", content=request.content, suggestions=None),
                assistant,
            ))
            await self.store.save(record)
            return TurnResponse(session_id=session_id, message=assistant.public())

    async def get(self, session_id: UUID7) -> SessionRecord:
        record = await self.store.load(session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return record

    async def delete(self, session_id: UUID7) -> None:
        async with self._lock(session_id):
            if not await self.store.delete(session_id):
                raise HTTPException(status_code=404, detail="Session not found")
