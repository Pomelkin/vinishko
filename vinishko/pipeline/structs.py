from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4

import numpy as np


def new_uuid() -> str:
    """Идентификатор бутылки: одна и та же строка у Candidate, его Sample и Rejection, в который он может превратиться."""
    return uuid4().hex


class Reason(StrEnum):
    """Основа для причин отказа. Каждый шаг пайплайна заводит свой наследник со своими причинами.

    Член задаётся тройкой: значение для логов и JSON, title — короткий текст для интерфейса,
    description — что случилось и почему это мешает распознаванию.
    """

    title: str
    description: str

    def __new__(cls, value: str, title: str, description: str) -> "Reason":
        """Член перечисления: строковое значение, заголовок для интерфейса и описание."""
        member = str.__new__(cls, value)
        member._value_ = value
        member.title = title
        member.description = description
        return member


@dataclass(frozen=True)
class Rejection:
    """Бутылка, которая дальше по пайплайну не идёт. Интерфейс показывает её маску и причину.

    Появляется на шаге нормализации либо позже из Candidate через Candidate.reject, тогда uuid сохраняется.
    Все координаты в пикселях оригинала после EXIF-поворота.
    """

    reason: Reason
    """Причина отказа из перечисления шага, который отказал; тексты для интерфейса — reason.title и reason.description."""
    detail: str
    """Что именно не прошло на этой бутылке: измеренное значение и порог."""
    score: float
    """Вероятность «целевая бутылка» по калибровке нормализации, от 0 до 1."""
    bottle: list
    """Маска бутылки: список полигонов, полигон — список точек [x, y]."""
    label: list | None
    """Маска этикетки, если она нашлась; None, если этикетка не искалась или не найдена."""
    uuid: str = field(default_factory=new_uuid)
    """Идентификатор бутылки; у отказа, полученного из Candidate, совпадает с его uuid."""

    @property
    def message(self) -> str:
        """Готовая строка для интерфейса: заголовок причины и что именно не прошло."""
        return f"{self.reason.title}: {self.detail}"


@dataclass(frozen=True)
class Candidate:
    """Годная бутылка: целевая, с читаемой основной этикеткой. Все координаты в пикселях оригинала после EXIF-поворота."""

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
    uuid: str = field(default_factory=new_uuid)
    """Идентификатор бутылки; по нему Sample и возможный Rejection связаны с этим кандидатом."""

    def reject(self, reason: Reason, detail: str) -> Rejection:
        """Отказ по этой бутылке на более позднем шаге пайплайна: те же маски, скор и uuid."""
        return Rejection(reason, detail, self.score, self.bottle, self.label, self.uuid)


@dataclass(eq=False)
class Sample:
    """Одна годная бутылка на пути по пайплайну: шаги дописывают сюда свои результаты.

    Создаётся нормализацией, один Sample на один Candidate. Новый шаг добавляет свои поля ниже со значением None по умолчанию.
    Шаг, который бракует бутылку, убирает Sample из потока и заменяет её Candidate в списке разметки на Rejection, см. reject.
    """

    candidate: Candidate
    """Бутылка, которой принадлежит всё остальное."""
    crop: np.ndarray
    """Нормализация: RGB-кроп для энкодера — поворот на candidate.angle, окно вокруг основной этикетки с запасом в пределах маски бутылки, фон вне маски залит серым."""
    crop_info: dict
    """Нормализация: как кроп получен из оригинала — угол, запас, матрицы matrix_src_to_dst и matrix_dst_to_src; см. render_bottle."""

    @property
    def uuid(self) -> str:
        """Идентификатор бутылки, тот же, что у candidate."""
        return self.candidate.uuid


def reject(
    samples: list[Sample],
    items: list[Candidate | Rejection],
    sample: Sample,
    reason: Reason,
    detail: str,
) -> None:
    """Шаг пайплайна бракует бутылку: Sample уходит из потока, её Candidate в разметке становится Rejection с тем же uuid."""
    samples.remove(sample)
    position = next(n for n, item in enumerate(items) if item.uuid == sample.uuid)
    items[position] = sample.candidate.reject(reason, detail)
