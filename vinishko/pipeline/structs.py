from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from typing import Literal
from uuid import uuid4

import numpy as np


Stage = Literal["normalization", "search", "resolve"]
"""Шаги пайплайна, которые могут отказать по бутылке."""


def new_uuid() -> str:
    """Идентификатор бутылки: одна и та же строка у BottleCrop и у RejectedBottle, в который он может превратиться."""
    return uuid4().hex


def polygons_bbox(polygons: list) -> list[int]:
    """Бокс [x1, y1, x2, y2] включительно по вершинам полигонов."""
    pts = np.concatenate(
        [np.asarray(p, np.float64).reshape(-1, 2) for p in polygons if len(p)]
    )
    (x1, y1), (x2, y2) = pts.min(0), pts.max(0)
    return [int(x1), int(y1), int(x2), int(y2)]


class Reason(StrEnum):
    """Основа для причин отказа. Каждый шаг пайплайна заводит свой наследник со своими причинами.

    Член задаётся тройкой: значение для логов и JSON, label — короткий текст для интерфейса (не title: у str это метод),
    description — что случилось и почему это мешает распознаванию. Наследник задаёт __stage__ = "<шаг>": имя шага, которому
    принадлежат его причины, так по отказу видно, кто отказал; dunder-имя, потому что любое другое Enum сделал бы членом.
    """

    label: str
    description: str
    __stage__: Stage

    @property
    def stage(self) -> Stage:
        """Шаг пайплайна, которому принадлежит причина."""
        return type(self).__stage__

    def __new__(cls, value: str, label: str, description: str) -> "Reason":
        """Член перечисления: строковое значение, заголовок для интерфейса и описание."""
        member = str.__new__(cls, value)
        member._value_ = value
        member.label = label
        member.description = description
        return member


@dataclass(frozen=True, slots=True)
class Rejection:
    """Отказ шага по бутылке: причина из перечисления шага и что именно не прошло. Шаг — reason.stage."""

    reason: Reason
    """Причина из перечисления шага; тексты для интерфейса — label и description."""
    detail: str = ""
    """Что именно не прошло на этой бутылке: измеренное значение и порог."""

    @property
    def stage(self) -> Stage:
        """Шаг, который отказал: normalization, search либо resolve."""
        return self.reason.stage

    @property
    def label(self) -> str:
        """Короткий текст причины для интерфейса."""
        return self.reason.label

    @property
    def description(self) -> str:
        """Что случилось и почему это мешает распознаванию."""
        return self.reason.description

    @property
    def message(self) -> str:
        """Готовая строка для интерфейса: заголовок причины и что именно не прошло."""
        return (
            f"{self.reason.label}: {self.detail}" if self.detail else self.reason.label
        )


@dataclass(frozen=True, slots=True)
class RejectedBottle:
    """Бутылка, по которой ответа не будет: не целевая, без читаемой этикетки, либо позже ничего не нашлось. Интерфейс показывает её маску и причину.

    Появляется на шаге нормализации либо позже из BottleCrop через RejectedBottle.of, тогда uuid сохраняется.
    Все координаты в пикселях оригинала после EXIF-поворота.
    """

    rejection: Rejection
    """Какой шаг отказал, почему и что именно не прошло."""
    score: float
    """Вероятность «целевая бутылка» по калибровке нормализации, от 0 до 1."""
    bottle: list
    """Маска бутылки: список полигонов, полигон — список точек [x, y]."""
    label: list | None
    """Маска этикетки, если она нашлась; None, если этикетка не искалась или не найдена."""
    uuid: str = field(default_factory=new_uuid)
    """Идентификатор бутылки; у отказа, полученного из BottleCrop, совпадает с его uuid."""

    @classmethod
    def of(cls, crop: "BottleCrop", rejection: Rejection) -> "RejectedBottle":
        """Отказ по годной бутылке на шаге после нормализации: те же маски, скор и uuid."""
        return cls(rejection, crop.score, crop.bottle, crop.label, crop.uuid)

    @property
    def message(self) -> str:
        """Готовая строка для интерфейса: заголовок причины и что именно не прошло."""
        return self.rejection.message


@dataclass(frozen=True, eq=False, slots=True)
class BottleCrop:
    """Годная бутылка с готовым кропом для энкодера: целевая, с читаемой основной этикеткой.

    Единица, которой оперируют шаги после нормализации: поиск получает список BottleCrop, реранкер — кандидатов Candidate
    с BottleCrop внутри. Маски и углы в пикселях оригинала после EXIF-поворота. Сравнение и хэш по идентичности объекта:
    внутри массив кропа.
    """

    index: int
    """Номер годной бутылки на картинке с 1, по убыванию score."""
    score: float
    """Вероятность «целевая бутылка» по калибровке нормализации, от 0 до 1."""
    bottle: list
    """Маска бутылки: список полигонов, полигон — список точек [x, y]."""
    label: list
    """Маска основной этикетки в том же формате; для мелкой бутылки уточнена вторым проходом SAM3."""
    angle: float
    """На сколько градусов повернуть кадр вокруг центра маски, чтобы бутылка встала горлышком вверх; аргумент cv2.getRotationMatrix2D.

    Равен 0, если виден короткий обрубок бутылки, обычно крупный план: ось у такой маски неустойчива, кадр не поворачивается.
    """
    crop: np.ndarray
    """RGB-кроп для визуального поиска: поворот на angle, окно вокруг основной этикетки с запасом в пределах маски бутылки, фон вне маски залит; см. render_bottle."""
    crop_info: dict
    """Как crop получен из оригинала — угол, запас, матрицы matrix_src_to_dst и matrix_dst_to_src; см. render_bottle."""
    box_crop: np.ndarray
    """RGB-кроп всей бутылки как на обычном фото: тот же поворот, bbox маски с небольшим запасом, фон не тронут; см. render_bottle_box.

    Идёт дальше по пайплайну, во второй уровень: VLM выбирает среди кандидатов по привычному фото. В каталоге позиции хранятся такими же кропами.
    """
    box_info: dict
    """Как box_crop получен из оригинала: угол, запас в пикселях, матрицы; см. render_bottle_box."""
    uuid: str = field(default_factory=new_uuid)
    """Идентификатор бутылки; по нему возможный RejectedBottle связан с этим кропом."""

    def markup(self) -> dict:
        """Разметка без кропа для JSON: index, score, bottle, label, angle, uuid — формат строк normalization.jsonl."""
        return {
            "index": self.index,
            "score": self.score,
            "bottle": self.bottle,
            "label": self.label,
            "angle": self.angle,
            "uuid": self.uuid,
        }


@dataclass(frozen=True, eq=False, slots=True)
class Candidate:
    """Позиция каталога, предложенная визуальным поиском для одной бутылки; единица работы второго уровня.

    Сравнение и хэш по идентичности объекта: внутри массив картинки.
    """

    slug: str
    """Идентификатор позиции каталога, он же ответ системы."""
    score: float
    """Косинус: собственный, если вектор позиции пришёл в выдаче, иначе косинус её группы — лучший из пришедших векторов группы."""
    image: np.ndarray
    """Картинка позиции из коллекции: вся бутылка каталожного фото по bbox маски, как box_crop у запроса — для второго уровня. Вектор считался с кропа поиска."""
    crop: BottleCrop
    """Бутылка запроса, для которой предложена позиция."""
    group: str
    """Группа одинакового дизайна, к которой относится позиция; у позиции без группы совпадает со slug."""
    retrieved: bool
    """Вектор позиции был в выдаче поиска, а не добавлен как член группы."""
    payload: dict
    """Метаданные точки коллекции: название, винодельня, винтаж и прочее из каталога — для второго уровня."""
    scores: dict[str, float]
    """Косинус по каждому входу энкодера, например {"crop": 0.91, "box_crop": 0.83}; score — их среднее. Пусто у позиции, добавленной как член группы."""


@dataclass(frozen=True, eq=False, slots=True)
class BottleCandidates:
    """Ответ визуального поиска по бутылке, для которой нашлись позиции каталога; список кандидатов не пуст.

    Ответов поиска, BottleCandidates либо UnmatchedBottle, ровно столько, сколько BottleCrop на входе, и в том же порядке.
    """

    crop: BottleCrop
    candidates: list[Candidate]

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError(
                "BottleCandidates без кандидатов; для бутылки без ответа — UnmatchedBottle"
            )


@dataclass(frozen=True, eq=False, slots=True)
class UnmatchedBottle:
    """Бутылка, для которой шаг не нашёл позиции каталога: поиск не нашёл похожих либо второй уровень отверг всех кандидатов.

    Кроп остаётся при бутылке, отказ — отдельно: какой шаг и почему, в rejection (SearchReason либо NearDuplicateReason).
    """

    crop: BottleCrop
    rejection: Rejection


@dataclass(frozen=True, eq=False, slots=True)
class MatchedBottle:
    """Ответ второго уровня: для бутылки выбрана ровно одна позиция каталога.

    Ответов второго уровня, MatchedBottle либо UnmatchedBottle, ровно столько, сколько BottleCandidates на входе, и в том же порядке.
    """

    crop: BottleCrop
    candidate: Candidate
    """Выбранная позиция; остальные кандидаты отброшены."""
    source: str
    """Какой вызов модели выбрал: group — внутри группы, других финалистов не было; final — среди лучших позиций групп и одиночек либо проверка единственной позиции."""
    checklist: dict
    """Наблюдения модели по производителю, профилю и году в её формате, из вызова, который выбрал позицию."""

    def __post_init__(self) -> None:
        if self.candidate.crop.uuid != self.crop.uuid:
            raise ValueError("позиция предложена для другой бутылки")


@dataclass(frozen=True, eq=False, slots=True)
class BottleOutcome:
    """Итог по одной бутылке фото после всех шагов: геометрия на исходном фото и ровно один исход.

    Исход — status: matched (выбрана одна позиция, match), candidates (второй уровень не запускался, кандидаты поиска как есть),
    rejected (отказ какого-то шага, rejection со stage), normalized (годная бутылка, поиск не запускался). Собирается оркестратором
    из выходов шагов по uuid бутылки.
    """

    uuid: str
    score: float
    """Скор отбора нормализации."""
    bottle: list
    """Маска бутылки на исходном фото: список полигонов, полигон — список точек [x, y]."""
    label: list | None
    """Маска этикетки, если нашлась."""
    crop: BottleCrop | None
    """Кропы бутылки; None, если нормализация отвергла."""
    match: MatchedBottle | None
    candidates: list[Candidate]
    rejection: Rejection | None

    def __post_init__(self) -> None:
        outcomes = sum(
            (self.match is not None, bool(self.candidates), self.rejection is not None)
        )
        if outcomes > 1 or (outcomes == 0 and self.crop is None):
            raise ValueError(
                "у бутылки не больше одного исхода: позиция, кандидаты либо отказ; без исхода только годная бутылка без поиска"
            )

    @property
    def status(self) -> Literal["matched", "candidates", "rejected", "normalized"]:
        """Исход бутылки."""
        if self.match is not None:
            return "matched"
        if self.candidates:
            return "candidates"
        return "rejected" if self.rejection is not None else "normalized"

    @property
    def bbox(self) -> list[int]:
        """Бокс маски [x1, y1, x2, y2] на исходном фото."""
        return polygons_bbox(self.bottle)
