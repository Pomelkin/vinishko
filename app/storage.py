"""Async disk storage with one atomic JSON file per session."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from pydantic import UUID7

from app.models import SessionRecord


class JsonSessionStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, session_id: UUID7) -> Path:
        return self.directory / f"{session_id}.json"

    async def load(self, session_id: UUID7) -> SessionRecord | None:
        return await asyncio.to_thread(self._load, session_id)

    def _load(self, session_id: UUID7) -> SessionRecord | None:
        try:
            raw = self._path(session_id).read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        record = SessionRecord.model_validate_json(raw)
        if record.session_id != session_id:
            raise ValueError("session file contains a different session ID")
        return record

    async def save(self, record: SessionRecord) -> None:
        await asyncio.to_thread(self._save, record)

    def _save(self, record: SessionRecord) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.directory,
                prefix=f".{record.session_id}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(record.model_dump_json())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self._path(record.session_id))
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    async def delete(self, session_id: UUID7) -> bool:
        return await asyncio.to_thread(self._delete, session_id)

    def _delete(self, session_id: UUID7) -> bool:
        try:
            self._path(session_id).unlink()
        except FileNotFoundError:
            return False
        return True
