from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from PIL import Image

from vinishko.pipeline.steps.normalization.normalize import Normalizer
from vinishko.pipeline.structs import BottleCrop, Candidate, RejectedBottle


class Searcher(Protocol):
    """Визуальный поиск: кропы бутылок → кандидаты каталога по каждому, в том же порядке; пустой список — ответа нет."""

    def __call__(self, crops: list[BottleCrop]) -> list[list[Candidate]]: ...


class Reranker(Protocol):
    """Уточнение внутри группы: те же списки кандидатов, переставленные и с пересчитанными скорами."""

    def __call__(self, candidates: list[list[Candidate]]) -> list[list[Candidate]]: ...


@dataclass
class PipelineResult:
    """Что пайплайн знает о картинке: все найденные бутылки и кандидаты по годным."""

    items: list[BottleCrop | RejectedBottle]
    """Разметка нормализации по убыванию скора отбора; бутылка, забракованная позже, стоит здесь RejectedBottle с тем же uuid."""
    candidates: list[list[Candidate]] = field(default_factory=list)
    """Кандидаты по каждому BottleCrop из items, порядок тот же."""

    @property
    def crops(self) -> list[BottleCrop]:
        """Годные бутылки."""
        return [item for item in self.items if isinstance(item, BottleCrop)]


def replace_rejected(
    items: list[BottleCrop | RejectedBottle], rejected: list[RejectedBottle]
) -> list[BottleCrop | RejectedBottle]:
    """Разметка, где бутылки с uuid из rejected заменены отказами: так шаг после нормализации бракует бутылку."""
    by_uuid = {item.uuid: item for item in rejected}
    return [by_uuid.get(item.uuid, item) for item in items]


class Pipeline:
    """Оркестратор: нормализация → визуальный поиск → реранкер. Шаги не знают друг о друге, их выходы связывает этот модуль.

    Шаги после нормализации подключаются по мере готовности: без поиска результат — одна разметка.
    """

    def __init__(
        self,
        normalizer: Normalizer,
        searcher: Searcher | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.normalizer, self.searcher, self.reranker = normalizer, searcher, reranker

    def __call__(self, img: Image.Image | Path | str) -> PipelineResult:
        """Картинка → разметка бутылок и кандидаты каталога по годным."""
        items = self.normalizer(img)
        crops = [item for item in items if isinstance(item, BottleCrop)]
        if not crops or self.searcher is None:
            return PipelineResult(items)
        candidates = self.searcher(crops)
        if self.reranker is not None:
            candidates = self.reranker(candidates)
        return PipelineResult(items, candidates)
