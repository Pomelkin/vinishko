"""Ручки сомелье в основном приложении: диалог по распознанному вину, состояние сессии хранится на диске."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Path as ApiPath
from fastapi import Request
from fastapi import Response
from pydantic import UUID7

from vinishko.sommelier.models import ErrorResponse
from vinishko.sommelier.models import OpenSessionRequest
from vinishko.sommelier.models import SessionResponse
from vinishko.sommelier.models import TurnResponse
from vinishko.sommelier.models import UserMessageRequest
from vinishko.sommelier.service import SommelierService


ERRORS = {
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    502: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}
SessionId = Annotated[
    UUID7, ApiPath(description="UUID версии 7, созданный клиентом для этой сессии.")
]

router = APIRouter(prefix="/sommelier", tags=["sommelier"])


def get_service(request: Request) -> SommelierService:
    return request.app.state.sommelier


Service = Annotated[SommelierService, Depends(get_service)]


@router.post(
    "/sessions/{session_id}",
    summary="Открыть сессию и получить первое сообщение",
    status_code=201,
    response_model=TurnResponse,
    responses={409: ERRORS[409], 502: ERRORS[502], 503: ERRORS[503]},
)
async def open_session(
    session_id: SessionId, request: OpenSessionRequest, svc: Service
) -> TurnResponse:
    return await svc.open(session_id, request)


@router.post(
    "/sessions/{session_id}/messages",
    summary="Отправить следующий вопрос",
    status_code=201,
    response_model=TurnResponse,
    responses={404: ERRORS[404], 502: ERRORS[502], 503: ERRORS[503]},
)
async def send_message(
    session_id: SessionId, request: UserMessageRequest, svc: Service
) -> TurnResponse:
    return await svc.ask(session_id, request)


@router.get(
    "/sessions/{session_id}",
    summary="Получить историю диалога",
    description=(
        "Возвращает все видимые сообщения сохранённой сессии по порядку: "
        "первый ответ сомелье, затем пары «вопрос пользователя — ответ сомелье». "
        "У первого ответа могут быть подсказки; у остальных сообщений suggestions равно null. "
        "Используйте метод для восстановления переписки в интерфейсе после обновления страницы. "
        "Для продолжения диалога вызывать его не требуется: POST /sommelier/sessions/{session_id}/messages сам использует "
        "сохранённую историю. Если сессия не найдена, возвращается 404."
    ),
    response_model=SessionResponse,
    responses={404: ERRORS[404]},
)
async def get_session(session_id: SessionId, svc: Service) -> SessionResponse:
    return (await svc.get(session_id)).public()


@router.delete(
    "/sessions/{session_id}",
    summary="Удалить сессию",
    status_code=204,
    responses={404: ERRORS[404]},
)
async def delete_session(session_id: SessionId, svc: Service) -> Response:
    await svc.delete(session_id)
    return Response(status_code=204)
