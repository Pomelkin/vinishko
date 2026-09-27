"""Оркестратор пайплайна: нормализация → визуальный поиск → реранкер. Шаги не знают друг о друге, их выходы связывает этот модуль."""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Protocol

from PIL import Image

from vinishko.pipeline.steps.normalization.normalize import Normalizer
from vinishko.pipeline.steps.near_duplicates import NearDuplicateReranker
from vinishko.pipeline.steps.vanilla_vlm_rerank import VanillaVlmReranker
from vinishko.pipeline.steps.vis_searcher.configs import TopNSearch, VisSearcherConfig
from vinishko.pipeline.steps.near_duplicates.rerank import NDR_CONCURRENCY
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

    def __init__(self, normalizer: Normalizer | None = None, searcher: Searcher | None = None, reranker: Reranker | None = None, *, enable_rerank: bool = True) -> None:
        self.normalizer = normalizer or Normalizer()
        self.searcher = searcher
        mode = os.environ.get("MATCH_MODE") or "groups"
        if mode not in {"groups", "vanilla_vlm"}:
            raise ValueError(f"MATCH_MODE должен быть groups или vanilla_vlm, получено {mode!r}")
        cfg = getattr(searcher, "cfg", None)
        if mode == "vanilla_vlm" and isinstance(cfg, VisSearcherConfig):
            cfg.top_k = 3
            cfg.search = TopNSearch(mode="top_n", cosine_threshold=-1.0)
        self.reranker = reranker if enable_rerank else None
        if enable_rerank and self.reranker is None and searcher is not None:
            chosen = VanillaVlmReranker if mode == "vanilla_vlm" else NearDuplicateReranker
            self.reranker = chosen.from_search_config(cfg)

    def __call__(self, img: Image.Image | Path | str) -> PipelineResult:
        """Картинка → выходы шагов: разметка нормализации, ответы поиска по годным бутылкам, время каждого шага."""
        return self.run_many([img])[0]

    def run_many(
        self,
        images: Sequence[Image.Image | Path | str],
        *,
        before_rerank: Callable[[int], None] | None = None,
        on_result: Callable[[int, PipelineResult], None] | None = None,
        run_rerank: bool = True,
    ) -> list[PipelineResult]:
        """Нормализует фото по одному, ищет все их кропы одним батчем, затем уточняет ответы по фото.

        Вызывающий ограничивает число фото в пачке, чтобы не держать много кропов в памяти.
        before_rerank получает индекс фото; CLI использует его для директории trace.
        Время общего поиска распределяется между фото пропорционально числу кропов.
        """
        emit = on_result or (lambda _index, _result: None)
        results: list[PipelineResult] = []
        crops_by_image: list[list[BottleCrop]] = []
        for img in images:
            started = time.perf_counter()
            normalization = self.normalizer(img)
            results.append(PipelineResult(normalization, timings={"normalization": time.perf_counter() - started}))
            crops_by_image.append([item for item in normalization if isinstance(item, BottleCrop)])

        crops = [crop for group in crops_by_image for crop in group]
        if not crops or self.searcher is None:
            for index, result in enumerate(results):
                emit(index, result)
            return results

        started = time.perf_counter()
        found = self.searcher(crops)
        search_time = time.perf_counter() - started
        if len(found) != len(crops):
            raise ValueError(f"поиск вернул {len(found)} ответов на {len(crops)} кропов")
        def timed_rerank(worker: Reranker, rows: list[BottleCandidates | UnmatchedBottle]) -> tuple[list[BottleCandidates | UnmatchedBottle], float]:
            started = time.perf_counter()
            return worker(rows), time.perf_counter() - started

        parallel = run_rerank and isinstance(self.reranker, NearDuplicateReranker)
        context = ThreadPoolExecutor(max_workers=NDR_CONCURRENCY) if parallel else nullcontext(None)
        with context as executor:
            jobs = {}
            offset = 0
            for index, (result, group) in enumerate(zip(results, crops_by_image, strict=True)):
                count = len(group)
                if not count:
                    if executor is None:
                        emit(index, result)
                    continue
                result.search = found[offset : offset + count]
                result.timings["search"] = search_time * count / len(crops)
                offset += count
                if run_rerank and self.reranker is not None:
                    if before_rerank is not None:
                        before_rerank(index)
                    if executor is not None:
                        worker = self.reranker.with_trace_dir(self.reranker.trace_dir)
                        jobs[index] = executor.submit(timed_rerank, worker, result.search)
                    else:
                        result.search, result.timings["rerank"] = timed_rerank(self.reranker, result.search)
                if executor is None:
                    emit(index, result)
            if executor is not None:
                for index, result in enumerate(results):
                    if index in jobs:
                        result.search, result.timings["rerank"] = jobs[index].result()
                    emit(index, result)
        return results
