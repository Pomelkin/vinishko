"""Public API and persisted session contracts."""

from __future__ import annotations

from typing import Annotated
from typing import Literal

from pydantic import UUID7
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import StringConstraints


CardValue = str | int | float | bool | None
CatalogCard = Annotated[
    dict[str, CardValue],
    Field(
        min_length=1,
        description="Каталожная карточка. Неиспользуемые поля допускаются, но игнорируются моделью.",
    ),
]
UserText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class OpenSessionRequest(StrictModel):
    wine: CatalogCard = Field(description="Распознанная основная карточка вина.")
    candidates: list[CatalogCard] = Field(
        max_length=5,
        description="До пяти альтернативных каталожных карточек, в порядке релевантности.",
    )


class UserMessageRequest(StrictModel):
    content: UserText = Field(
        description="Вопрос пользователя или текст выбранной подсказки."
    )


class PublicMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str
    suggestions: list[str] | None = Field(
        description="Подсказки первого ответа, если модель их вернула; в следующих ответах null.",
    )


class TurnResponse(StrictModel):
    session_id: UUID7
    message: PublicMessage


class SessionResponse(StrictModel):
    session_id: UUID7
    messages: list[PublicMessage]


class ErrorResponse(StrictModel):
    detail: str


class StoredMessage(PublicMessage):
    # The raw provider message is required for reasoning continuity.
    provider_message: dict[str, object] | None = None

    def public(self) -> PublicMessage:
        return PublicMessage.model_validate(
            self.model_dump(exclude={"provider_message"})
        )


class SessionRecord(StrictModel):
    session_id: UUID7
    wine: CatalogCard
    candidates: list[CatalogCard] = Field(max_length=5)
    messages: list[StoredMessage]

    def public(self) -> SessionResponse:
        return SessionResponse(
            session_id=self.session_id,
            messages=[message.public() for message in self.messages],
        )

    def provider_history(self, user_text: str) -> list[dict[str, object]]:
        history: list[dict[str, object]] = []
        for message in self.messages:
            if message.role == "user":
                history.append({"role": "user", "content": message.content})
            else:
                if message.provider_message is None:
                    raise ValueError("stored assistant message has no provider payload")
                history.append(message.provider_message)
        history.append({"role": "user", "content": user_text})
        return history
