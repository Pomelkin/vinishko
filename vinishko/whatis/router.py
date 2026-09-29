"""Ручка whatis в основном приложении: категория и винодельня по фото бутылки, которой нет в каталоге."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import HTTPException
from fastapi import Request
from fastapi import UploadFile

from vinishko.whatis.models import ErrorResponse
from vinishko.whatis.models import RecognitionResponse
from vinishko.whatis.service import RecognitionService


MAX_IMAGE_BYTES = 10 * 1024 * 1024

router = APIRouter(tags=["whatis"])


def get_service(request: Request) -> RecognitionService:
    return request.app.state.whatis


Service = Annotated[RecognitionService, Depends(get_service)]


@router.post(
    "/whatis",
    summary="Определить категорию и винодельню",
    response_model=RecognitionResponse,
    responses={
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def whatis(
    image: Annotated[
        UploadFile,
        File(description="Фото бутылки: JPEG, PNG, WebP или GIF, до 10 МиБ."),
    ],
    svc: Service,
) -> RecognitionResponse:
    data = await image.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MiB")
    return await svc.recognize(data)
