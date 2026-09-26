from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4

import numpy as np


def new_uuid() -> str:
    """Идентификатор бутылки: одна и та же строка у BottleCrop и у RejectedBottle, в который он может превратиться."""
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
class RejectedBottle:
    """Бутылка, по которой ответа не будет: не целевая, без читаемой этикетки, либо позже ничего не нашлось. Интерфейс показывает её маску и причину.

    Появляется на шаге нормализации либо позже из BottleCrop через BottleCrop.reject, тогда uuid сохраняется.
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
    """Идентификатор бутылки; у отказа, полученного из BottleCrop, совпадает с его uuid."""

    @property
    def message(self) -> str:
        """Готовая строка для интерфейса: заголовок причины и что именно не прошло."""
        return f"{self.reason.title}: {self.detail}"


@dataclass(frozen=True, eq=False)
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
    """RGB-кроп для энкодера: поворот на angle, окно вокруг основной этикетки с запасом в пределах маски бутылки, фон вне маски залит; см. render_bottle."""
    crop_info: dict
    """Как кроп получен из оригинала — угол, запас, матрицы matrix_src_to_dst и matrix_dst_to_src; см. render_bottle."""
    uuid: str = field(default_factory=new_uuid)
    """Идентификатор бутылки; по нему возможный RejectedBottle связан с этим кропом."""

    def reject(self, reason: Reason, detail: str) -> RejectedBottle:
        """Отказ по этой бутылке на более позднем шаге пайплайна: те же маски, скор и uuid."""
        return RejectedBottle(reason, detail, self.score, self.bottle, self.label, self.uuid)

    def markup(self) -> dict:
        """Разметка без кропа для JSON: index, score, bottle, label, angle, uuid — формат строк normalization.jsonl."""
        return {"index": self.index, "score": self.score, "bottle": self.bottle, "label": self.label, "angle": self.angle, "uuid": self.uuid}


@dataclass(frozen=True, eq=False)
class Candidate:
    """Позиция каталога, предложенная визуальным поиском для одной бутылки; единица работы второго уровня.

    Сравнение и хэш по идентичности объекта: внутри массив картинки.
    """

    slug: str
    """Идентификатор позиции каталога, он же ответ системы."""
    score: float
    """Косинус: собственный, если вектор позиции пришёл в выдаче, иначе косинус её группы — лучший из пришедших векторов группы."""
    image: np.ndarray
    """RGB-кроп позиции из коллекции, с которого считался её вектор: нужен верификации второго уровня."""
    crop: BottleCrop
    """Бутылка запроса, для которой предложена позиция."""
    group: str
    """Группа одинакового дизайна, к которой относится позиция; у позиции без группы совпадает со slug."""
    retrieved: bool
    """Вектор позиции был в выдаче поиска, а не добавлен как член группы."""
    payload: dict
    """Метаданные точки коллекции: название, винодельня, винтаж и прочее из каталога — для второго уровня."""
