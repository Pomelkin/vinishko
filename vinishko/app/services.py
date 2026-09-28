"""HTTP-маршруты к отдельным сервисам сомелье и неизвестного вина."""

from typing import Annotated

import httpx
from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from fastapi import Path as ApiPath
from pydantic import BaseModel, UUID7, ValidationError

from vinishko.app.schemas import OpenSessionRequest, SessionResponse, TurnResponse
from vinishko.app.schemas import UnknownWineOut, UserMessageRequest


SOMMELIER_URL = "http://127.0.0.1:8805"
WHATIS_URL = "http://127.0.0.1:8810"
SERVICE_TIMEOUT_SECONDS = 65.0
MAX_IMAGE_BYTES = 10 * 1024 * 1024

router = APIRouter()
SessionId = Annotated[UUID7, ApiPath(description="UUIDv7, созданный клиентом")]


async def service_request[T: BaseModel](
    client: httpx.AsyncClient,
    method: str,
    url: str,
    response_model: type[T] | None,
    expected_status: int = 200,
    **kwargs,
) -> T | None:
    """Сохраняет ошибки сервиса и проверяет его публичный ответ."""
    try:
        response = await client.request(method, url, **kwargs)
    except httpx.TimeoutException as error:
        raise HTTPException(status_code=504, detail="Истекло время ожидания микросервиса") from error
    except httpx.ConnectError as error:
        endpoint = httpx.URL(url)
        raise HTTPException(status_code=503, detail=f"Микросервис недоступен на {endpoint.host}:{endpoint.port}. Запустите сервис по инструкции в vinishko/app/README.md") from error
    except httpx.RequestError as error:
        raise HTTPException(status_code=502, detail="Ошибка связи с микросервисом") from error
    if response.is_error:
        try:
            detail = response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        raise HTTPException(
            status_code=response.status_code,
            detail=detail if isinstance(detail, (str, list)) else "Ошибка микросервиса",
        )
    if response.status_code != expected_status:
        raise HTTPException(status_code=502, detail="Неожиданный статус микросервиса")
    if response_model is None:
        return None
    try:
        return response_model.model_validate_json(response.content)
    except ValidationError as error:
        raise HTTPException(status_code=502, detail="Невалидный ответ микросервиса") from error


async def predict_unknown(client: httpx.AsyncClient, data: bytes) -> UnknownWineOut:
    """Фото одной бутылки → категория и винодельня."""
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MiB")
    return await service_request(
        client, "POST", f"{WHATIS_URL}/v1/predict", UnknownWineOut,
        files={"image": ("bottle.jpg", data, "application/octet-stream")},
    )


@router.post("/v1/predict", response_model=UnknownWineOut, tags=["unknown wine"])
async def predict(
    request: Request,
    image: Annotated[UploadFile, File(description="JPEG, PNG, WebP или GIF, до 10 МиБ")],
) -> UnknownWineOut:
    return await predict_unknown(request.app.state.services, await image.read(MAX_IMAGE_BYTES + 1))


@router.post("/v1/sessions/{session_id}", response_model=TurnResponse, status_code=201, tags=["sommelier"])
async def open_session(request: Request, session_id: SessionId, body: OpenSessionRequest) -> TurnResponse:
    return await service_request(
        request.app.state.services, "POST", f"{SOMMELIER_URL}/v1/sessions/{session_id}",
        TurnResponse, 201, json=body.model_dump(mode="json"),
    )


@router.post("/v1/sessions/{session_id}/messages", response_model=TurnResponse, status_code=201, tags=["sommelier"])
async def send_message(request: Request, session_id: SessionId, body: UserMessageRequest) -> TurnResponse:
    return await service_request(
        request.app.state.services, "POST", f"{SOMMELIER_URL}/v1/sessions/{session_id}/messages",
        TurnResponse, 201, json=body.model_dump(mode="json"),
    )


@router.get("/v1/sessions/{session_id}", response_model=SessionResponse, tags=["sommelier"])
async def get_session(request: Request, session_id: SessionId) -> SessionResponse:
    return await service_request(
        request.app.state.services, "GET", f"{SOMMELIER_URL}/v1/sessions/{session_id}", SessionResponse,
    )


@router.delete("/v1/sessions/{session_id}", status_code=204, tags=["sommelier"])
async def delete_session(request: Request, session_id: SessionId) -> Response:
    await service_request(
        request.app.state.services, "DELETE", f"{SOMMELIER_URL}/v1/sessions/{session_id}", None, 204,
    )
    return Response(status_code=204)
