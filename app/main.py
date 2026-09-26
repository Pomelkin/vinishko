"""FastAPI entry point and generated OpenAPI specification."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile

from app.models import ErrorResponse, RecognitionResponse
from app.service import RecognitionService


MAX_IMAGE_BYTES = 10 * 1024 * 1024


def get_service(request: Request) -> RecognitionService:
    return request.app.state.service


Service = Annotated[RecognitionService, Depends(get_service)]


def create_app(service: RecognitionService | None = None) -> FastAPI:
    app = FastAPI(
        title="Vinishko Unknown Wine Recognition API",
        version="1.0.0",
        description="Определяет категорию вина и винодельню по фото бутылки, отсутствующей в каталоге.",
    )
    app.state.service = service if service is not None else RecognitionService()

    @app.get("/health", tags=["service"], summary="Проверка процесса")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/v1/predict",
        tags=["recognition"],
        summary="Определить категорию и винодельню",
        response_model=RecognitionResponse,
        responses={
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def recognize(
        image: Annotated[UploadFile, File(description="Фото бутылки: JPEG, PNG, WebP или GIF, до 10 МиБ.")],
        svc: Service,
    ) -> RecognitionResponse:
        data = await image.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds 10 MiB")
        return await svc.recognize(data)

    return app


app = create_app()
