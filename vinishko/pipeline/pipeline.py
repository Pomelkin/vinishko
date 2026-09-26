from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from PIL import Image

from vinishko.pipeline.steps.normalization.normalize import Normalizer
from vinishko.pipeline.structs import BottleCrop, RejectedBottle, SearchResult


class Searcher(Protocol):
    """Визуальный поиск: кропы бутылок → SearchResult на каждый, в том же порядке: кандидаты каталога либо отказ с причиной."""

    def __call__(self, crops: list[BottleCrop]) -> list[SearchResult]: ...


class Reranker(Protocol):
    """Уточнение внутри группы: те же результаты с переставленными и пересчитанными кандидатами."""

    def __call__(self, results: list[SearchResult]) -> list[SearchResult]: ...


@dataclass(slots=True)
class PipelineResult:
    """Что пайплайн знает о картинке: все найденные бутылки и ответ поиска по каждой годной."""

    items: list[BottleCrop | RejectedBottle]
    """Разметка нормализации по убыванию скора отбора; бутылка, забракованная позже, стоит здесь RejectedBottle с тем же uuid."""
    results: list[SearchResult] = field(default_factory=list)
    """По одному на каждую годную бутылку нормализации, в том же порядке; бутылка без ответа здесь с причиной, а в items уже RejectedBottle."""

    @property
    def crops(self) -> list[BottleCrop]:
        """Бутылки, оставшиеся годными после всех шагов."""
        return [item for item in self.items if isinstance(item, BottleCrop)]


def replace_rejected(
    items: list[BottleCrop | RejectedBottle], rejected: list[RejectedBottle]
) -> list[BottleCrop | RejectedBottle]:
    """Разметка, где бутылки с uuid из rejected заменены отказами: так шаг после нормализации бракует бутылку."""
    by_uuid = {item.uuid: item for item in rejected}
    return [by_uuid.get(item.uuid, item) for item in items]


class Pipeline:
    """Оркестратор: нормализация → визуальный поиск → реранкер. Шаги не знают друг о друге, их выходы связывает этот модуль.

    Без normalizer поднимается Normalizer() с конфигом и устройством по умолчанию. Шаги после нормализации подключаются по мере готовности:
    без поиска результат — одна разметка.
    """

    def __init__(
        self,
        normalizer: Normalizer | None = None,
        searcher: Searcher | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.normalizer = normalizer or Normalizer()
        self.searcher, self.reranker = searcher, reranker

    def __call__(self, img: Image.Image | Path | str) -> PipelineResult:
        """Картинка → разметка бутылок и ответ поиска по каждой годной; бутылка без ответа становится в разметке отказом с причиной."""
        items = self.normalizer(img)
        crops = [item for item in items if isinstance(item, BottleCrop)]
        if not crops or self.searcher is None:
            return PipelineResult(items)
        results = self.searcher(crops)
        if self.reranker is not None:
            results = self.reranker(results)
        items = replace_rejected(
            items, [r.rejected for r in results if r.rejected is not None]
        )
        return PipelineResult(items, results)
