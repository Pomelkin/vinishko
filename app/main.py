"""FastAPI entry point and generated OpenAPI specification."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Path as ApiPath, Request, Response
from pydantic import UUID7

from app.models import (
    ErrorResponse,
    OpenSessionRequest,
    SessionResponse,
    TurnResponse,
    UserMessageRequest,
)
from app.service import SommelierService
from app.storage import JsonSessionStore


ERRORS = {404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}}
SessionId = Annotated[UUID7, ApiPath(description="UUID версии 7, созданный клиентом для этой сессии.")]


def get_service(request: Request) -> SommelierService:
    return request.app.state.service


Service = Annotated[SommelierService, Depends(get_service)]


def create_app(service: SommelierService | None = None) -> FastAPI:
    app = FastAPI(
        title="Vinishko AI Sommelier API",
        version="1.0.0",
        description="Диалог с AI-сомелье по распознанному вину. Состояние сессии хранится в сервисе.",
    )
    if service is None:
        directory = Path(os.environ.get("SOMMELIER_DATA_DIR", "data/sessions")).resolve()
        service = SommelierService(JsonSessionStore(directory))
    app.state.service = service

    @app.get("/health", tags=["service"], summary="Проверка процесса")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/v1/sessions/{session_id}",
        tags=["sessions"],
        summary="Открыть сессию и получить первое сообщение",
        status_code=201,
        response_model=TurnResponse,
        responses={409: ERRORS[409], 502: ERRORS[502], 503: ERRORS[503]},
    )
    async def open_session(session_id: SessionId, request: OpenSessionRequest, svc: Service) -> TurnResponse:
        return await svc.open(session_id, request)

    @app.post(
        "/v1/sessions/{session_id}/messages",
        tags=["sessions"],
        summary="Отправить следующий вопрос",
        status_code=201,
        response_model=TurnResponse,
        responses={404: ERRORS[404], 502: ERRORS[502], 503: ERRORS[503]},
    )
    async def send_message(session_id: SessionId, request: UserMessageRequest, svc: Service) -> TurnResponse:
        return await svc.ask(session_id, request)

    @app.get(
        "/v1/sessions/{session_id}",
        tags=["sessions"],
        summary="Получить историю диалога",
        response_model=SessionResponse,
        responses={404: ERRORS[404]},
    )
    async def get_session(session_id: SessionId, svc: Service) -> SessionResponse:
        return (await svc.get(session_id)).public()

    @app.delete(
        "/v1/sessions/{session_id}",
        tags=["sessions"],
        summary="Удалить сессию",
        status_code=204,
        responses={404: ERRORS[404]},
    )
    async def delete_session(session_id: SessionId, svc: Service) -> Response:
        await svc.delete(session_id)
        return Response(status_code=204)

    return app


app = create_app()
