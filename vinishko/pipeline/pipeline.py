"""Оркестратор пайплайна: нормализация → визуальный поиск → реранкер. Шаги не знают друг о друге, их выходы связывает этот модуль."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from PIL import Image

from vinishko.pipeline.steps.normalization.normalize import Normalizer
from vinishko.pipeline.structs import BottleCandidates, BottleCrop, RejectedBottle, UnmatchedBottle


class Searcher(Protocol):
    """Визуальный поиск: кропы бутылок → ответ на каждый, в том же порядке: BottleCandidates с кандидатами каталога либо UnmatchedBottle с отказом и причиной."""

    def __call__(self, crops: list[BottleCrop]) -> list[BottleCandidates | UnmatchedBottle]: ...


class Reranker(Protocol):
    """Уточнение внутри группы: те же ответы с переставленными и пересчитанными кандидатами."""

    def __call__(self, results: list[BottleCandidates | UnmatchedBottle]) -> list[BottleCandidates | UnmatchedBottle]: ...


def replace_rejected(items: list[BottleCrop | RejectedBottle], rejected: list[RejectedBottle]) -> list[BottleCrop | RejectedBottle]:
    """Разметка, где бутылки с uuid из rejected заменены отказами: так шаг после нормализации бракует бутылку."""
    by_uuid = {item.uuid: item for item in rejected}
    return [by_uuid.get(item.uuid, item) for item in items]


@dataclass(slots=True)
class PipelineResult:
    """Выходы шагов по одной картинке."""

    normalization: list[BottleCrop | RejectedBottle]
    """Разметка нормализации как есть, по убыванию скора отбора."""
    search: list[BottleCandidates | UnmatchedBottle] = field(default_factory=list)
    """Ответ поиска по каждому BottleCrop нормализации, в том же порядке; пусто, если поиск не запускался или годных бутылок нет."""
    timings: dict[str, float] = field(default_factory=dict)
    """Секунды на шаг."""

    @property
    def items(self) -> list[BottleCrop | RejectedBottle]:
        """Итоговая разметка: бутылка без ответа поиска стоит здесь RejectedBottle с тем же uuid и причиной шага поиска."""
        return replace_rejected(self.normalization, [r.rejected for r in self.search if isinstance(r, UnmatchedBottle)])

    @property
    def crops(self) -> list[BottleCrop]:
        """Бутылки, оставшиеся годными после всех шагов."""
        return [item for item in self.items if isinstance(item, BottleCrop)]


class Pipeline:
    """Оркестратор: нормализация → визуальный поиск → реранкер.

    Без normalizer поднимается Normalizer() с конфигом и устройством по умолчанию. Шаги после нормализации подключаются по мере
    готовности: без поиска результат — одна разметка.
    """

    def __init__(self, normalizer: Normalizer | None = None, searcher: Searcher | None = None, reranker: Reranker | None = None) -> None:
        self.normalizer = normalizer or Normalizer()
        self.searcher, self.reranker = searcher, reranker

    def __call__(self, img: Image.Image | Path | str) -> PipelineResult:
        """Картинка → выходы шагов: разметка нормализации, ответы поиска по годным бутылкам, время каждого шага."""
        started = time.perf_counter()
        normalization = self.normalizer(img)
        result = PipelineResult(normalization, timings={"normalization": time.perf_counter() - started})
        crops = [item for item in normalization if isinstance(item, BottleCrop)]
        if not crops or self.searcher is None:
            return result
        started = time.perf_counter()
        result.search = self.searcher(crops)
        result.timings["search"] = time.perf_counter() - started
        if self.reranker is not None:
            started = time.perf_counter()
            result.search = self.reranker(result.search)
            result.timings["rerank"] = time.perf_counter() - started
        return result
