"""Схемы ответов API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, UUID7
from pydantic import Field


Polygon = list[list[float]]
"""Полигон маски: список точек [x, y] в пикселях исходного фото после EXIF-поворота."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class UnknownWineOut(StrictModel):
    category: str
    brand: str


class ServiceErrorOut(BaseModel):
    status_code: int
    detail: str


CatalogCard = Annotated[dict[str, str | int | float | bool | None], Field(min_length=1)]


class OpenSessionRequest(StrictModel):
    wine: CatalogCard
    candidates: list[CatalogCard] = Field(max_length=5)


class UserMessageRequest(StrictModel):
    content: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class PublicMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str
    suggestions: list[str] | None


class TurnResponse(StrictModel):
    session_id: UUID7
    message: PublicMessage


class SessionResponse(StrictModel):
    session_id: UUID7
    messages: list[PublicMessage]


class RejectionOut(BaseModel):
    """Почему по бутылке нет ответа."""

    stage: str = Field(
        description="Шаг, который отказал: search либо resolve; отказы normalization в ответ не попадают"
    )
    reason: str = Field(description="Код причины из перечисления шага")
    label: str = Field(description="Короткий текст причины для интерфейса")
    description: str = Field(
        description="Что случилось и почему это мешает распознаванию"
    )
    detail: str = Field(
        description="Что именно не прошло на этой бутылке: измеренное значение и порог"
    )
    message: str = Field(description="label и detail одной строкой")


class CandidateOut(BaseModel):
    """Позиция каталога с визуальным скором: кандидат поиска, когда второй уровень выключен; от неё же MatchOut."""

    slug: str
    score: float = Field(
        description="Скор визуального поиска: среднее косинусов по входам энкодера"
    )
    group: str = Field(description="Группа одинакового дизайна")
    image_url: str = Field(
        description="Картинка позиции из коллекции, относительный путь на этом же сервере"
    )
    catalog: dict[str, str] = Field(
        description="Строка каталога, прочитанного при старте: все колонки CSV, ключи — заголовки"
    )


class MatchOut(CandidateOut):
    """Выбранная позиция каталога."""

    source: str = Field(
        description="ndr_v5 — модель выбрала среди группы; vector — в группе одна позиция, модель не вызывалась"
    )
    checklist: dict = Field(
        default_factory=dict,
        description="Наблюдения модели второго уровня по производителю, профилю и году; пусто без модели",
    )


class BottleOut(BaseModel):
    """Бутылка на фото и её исход."""

    uuid: str
    index: int = Field(description="Номер годной бутылки с 1 по убыванию скора отбора")
    score: float = Field(
        description="Скор отбора нормализации, вероятность целевой бутылки"
    )
    polygons: list[Polygon] = Field(description="Маска бутылки на исходном фото")
    label_polygons: list[Polygon] | None = Field(
        description="Маска этикетки, если нашлась"
    )
    bbox: list[int] = Field(description="[x1, y1, x2, y2] маски бутылки")
    status: str = Field(
        description="matched — позиция выбрана; rejected — отказ поиска или второго уровня; candidates — кандидаты поиска без второго уровня"
    )
    match: MatchOut | None = None
    candidates: list[CandidateOut] = Field(default_factory=list)
    rejection: RejectionOut | None = None
    unknown_wine: UnknownWineOut | None = Field(default=None, description="Категория и винодельня от whatis для отвергнутой бутылки")
    unknown_wine_error: ServiceErrorOut | None = Field(default=None, description="Ошибка whatis; исход распознавания сохранён")


class ImageOut(BaseModel):
    """Размер фото после EXIF-поворота: в этих координатах маски."""

    width: int
    height: int


class RecognizeResponse(BaseModel):
    """Ответ на фото: бутылки по убыванию скора отбора.

    В списке только бутылки, дошедшие до поиска: с выбранной позицией либо с отказом поиска или второго уровня, у всех маска.
    Объекты, которые нормализация не сочла целевой бутылкой с читаемой этикеткой, в список не попадают, их число — ignored.
    """

    image: ImageOut
    bottles: list[BottleOut]
    ignored: int = Field(
        description="Сколько объектов на фото нормализация отбросила: не целевая бутылка, нет читаемой этикетки, лишняя при лимите"
    )
    timings_s: dict[str, float]


class HealthResponse(BaseModel):
    """Состояние сервиса."""

    status: str
    collection: str | None
    model: str | None
    encoder_input: list[str]
    search_mode: str | None
    resolver: str | None = Field(
        description="Модель второго уровня; None, если он выключен"
    )
    catalog: str
    catalog_rows: int
