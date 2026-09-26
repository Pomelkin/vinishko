"""HTTP-facing orchestration around the copied synchronous predictor."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError

from app.models import RecognitionResponse
from app.solution import predict
from app.solution.catalog import UNKNOWN, catalog_values
from app.solution.predictor import MissingApiKeyError


logger = logging.getLogger(__name__)
RecognitionEngine = Callable[[bytes], dict[str, Any]]


class RecognitionService:
    def __init__(self, engine: RecognitionEngine = predict) -> None:
        self.engine = engine

    async def recognize(self, image: bytes) -> RecognitionResponse:
        try:
            result = await asyncio.to_thread(self.engine, image)
        except MissingApiKeyError:
            raise HTTPException(status_code=503, detail="OpenRouter is not configured") from None
        except ValueError as error:
            if str(error) == "Неподдерживаемая сигнатура изображения":
                raise HTTPException(status_code=415, detail="Unsupported image format") from None
            logger.exception("Recognition configuration or input failed")
            raise HTTPException(status_code=502, detail="Recognition is temporarily unavailable") from None
        except Exception:
            logger.exception("Recognition engine crashed")
            raise HTTPException(status_code=502, detail="Recognition is temporarily unavailable") from None

        if not isinstance(result, dict) or result.get("_status") != "ok":
            logger.error("Recognition failed: %s: %s", result.get("_status") if isinstance(result, dict) else "invalid_result", result.get("_error") if isinstance(result, dict) else "engine returned non-object")
            raise HTTPException(status_code=502, detail="Recognition is temporarily unavailable")

        try:
            response = RecognitionResponse.model_validate({
                "category": result.get("category"),
                "brand": result.get("brand"),
            })
        except ValidationError:
            raise HTTPException(status_code=502, detail="Recognition returned an invalid response") from None
        categories, brands = catalog_values()
        if response.category not in (*categories, UNKNOWN) or response.brand not in (*brands, UNKNOWN):
            raise HTTPException(status_code=502, detail="Recognition returned an invalid response")
        return response
