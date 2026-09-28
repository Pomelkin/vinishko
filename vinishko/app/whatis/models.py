"""Public recognition response contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RecognitionResponse(StrictModel):
    category: str = Field(description="Цвет вина из каталога либо «Не удалось определить».")
    brand: str = Field(description="Каноническая винодельня либо «Не удалось определить».")


class ErrorResponse(StrictModel):
    detail: str
