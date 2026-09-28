"""Целое фото → SigLIP2 → Qdrant. Один запрос и один ответ на фото."""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from vinishko.pipeline.structs import BottleCandidates, BottleCrop, RejectedBottle, UnmatchedBottle

from .steps.vis_searcher import VisSearcher


class Searcher(Protocol):
    def __call__(self, crops: list[BottleCrop]) -> list[BottleCandidates | UnmatchedBottle]: ...


def raw_crop(image: Image.Image | Path | str) -> BottleCrop:
    """Как DenseRetriever: convert RGB, целый кадр без сегментации и поворота."""
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB")).copy()
    else:
        with Image.open(image) as opened:
            rgb = np.asarray(opened.convert("RGB")).copy()
    height, width = rgb.shape[:2]
    whole = [[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]]
    info = {"mode": "raw", "width": width, "height": height}
    return BottleCrop(1, 0.0, whole, [], 0.0, rgb, info, rgb, info)


@dataclass(slots=True)
class PipelineResult:
    """Прежний формат результатов; normalization содержит исходный кадр для совместимости."""

    normalization: list[BottleCrop | RejectedBottle]
    search: list[BottleCandidates | UnmatchedBottle] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def items(self) -> list[BottleCrop | RejectedBottle]:
        rejected = {r.rejected.uuid: r.rejected for r in self.search if isinstance(r, UnmatchedBottle)}
        return [rejected.get(item.uuid, item) for item in self.normalization]

    @property
    def crops(self) -> list[BottleCrop]:
        return [item for item in self.items if isinstance(item, BottleCrop)]


class Pipeline:
    def __init__(self, searcher: Searcher | None = None) -> None:
        self.searcher = searcher if searcher is not None else VisSearcher()

    def __call__(self, img: Image.Image | Path | str) -> PipelineResult:
        return self.run_many([img])[0]

    def run_many(self, images: Sequence[Image.Image | Path | str], *,
                 on_result: Callable[[int, PipelineResult], None] | None = None) -> list[PipelineResult]:
        results = []
        crops: list[BottleCrop] = []
        for image in images:
            started = time.perf_counter()
            crop = raw_crop(image)
            crops.append(crop)
            results.append(PipelineResult([crop], timings={"raw_input": time.perf_counter() - started}))
        if not results:
            return []
        started = time.perf_counter()
        found = self.searcher(crops)
        elapsed = time.perf_counter() - started
        if len(found) != len(crops):
            raise ValueError(f"поиск вернул {len(found)} ответов на {len(crops)} фото")
        for index, (result, answer, crop) in enumerate(zip(results, found, crops, strict=True)):
            if answer.crop.uuid != crop.uuid:
                raise ValueError("поиск нарушил порядок ответов")
            result.search = [answer]
            result.timings["search"] = elapsed / len(crops)
            if on_result is not None:
                on_result(index, result)
        return results
