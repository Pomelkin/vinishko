"""Оркестратор пайплайна: нормализация → визуальный поиск → выбор позиции либо отказ. Шаги не знают друг о друге, их выходы связывает этот модуль."""

import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Protocol

from PIL import Image

from vinishko.pipeline.catalog import Catalog
from vinishko.pipeline.catalog import load_catalog
from vinishko.pipeline.configs import load_config
from vinishko.pipeline.steps.filter import FilterResolver
from vinishko.pipeline.steps.filter.resolve import cards_from_rows as filter_cards_from_rows
from vinishko.pipeline.steps.near_duplicates import NearDuplicateResolver
from vinishko.pipeline.steps.near_duplicates.resolve import cards_from_rows
from vinishko.pipeline.steps.normalization.normalize import Normalizer
from vinishko.pipeline.steps.vis_searcher import VisSearcher
from vinishko.pipeline.steps.vis_searcher.configs import TopNSearch
from vinishko.pipeline.structs import BottleCandidates
from vinishko.pipeline.structs import BottleCrop
from vinishko.pipeline.structs import BottleOutcome
from vinishko.pipeline.structs import MatchedBottle
from vinishko.pipeline.structs import RejectedBottle
from vinishko.pipeline.structs import UnmatchedBottle


class Searcher(Protocol):
    """Визуальный поиск: кропы бутылок → ответ на каждый, в том же порядке: BottleCandidates с кандидатами каталога либо UnmatchedBottle с отказом и причиной."""

    def __call__(
        self, crops: list[BottleCrop]
    ) -> list[BottleCandidates | UnmatchedBottle]: ...

    def check_normalization(self, normalization: dict) -> None:
        """Ошибка, если нормализатор с таким crop_config вырезает кропы не так, как при сборке каталога поиска."""
        ...

    def slugs(self) -> set[str]:
        """Все позиции, которые поиск может выдать кандидатами."""
        ...


class Resolver(Protocol):
    """Второй уровень: по кандидатам каждой бутылки выбирает ровно одну позицию каталога либо отказывает.

    Получает только бутылки с кандидатами; ответов столько же и в том же порядке: MatchedBottle с одной позицией либо UnmatchedBottle с причиной.
    """

    def __call__(
        self, found: list[BottleCandidates]
    ) -> list[MatchedBottle | UnmatchedBottle]: ...


def replace_rejected(
    items: list[BottleCrop | RejectedBottle], rejected: list[RejectedBottle]
) -> list[BottleCrop | RejectedBottle]:
    """Разметка, где бутылки с uuid из rejected заменены отказами: так шаг после нормализации бракует бутылку."""
    by_uuid = {item.uuid: item for item in rejected}
    return [by_uuid.get(item.uuid, item) for item in items]


@dataclass(slots=True)
class PipelineResult:
    """Выходы шагов по одной картинке и собранный из них итог по каждой бутылке."""

    normalization: list[BottleCrop | RejectedBottle]
    """Разметка нормализации как есть, по убыванию скора отбора."""
    search: list[BottleCandidates | UnmatchedBottle] = field(default_factory=list)
    """Ответ поиска по каждому BottleCrop нормализации, в том же порядке; пусто, если поиск не запускался или годных бутылок нет."""
    resolution: list[MatchedBottle | UnmatchedBottle] = field(default_factory=list)
    """Ответ второго уровня по каждому элементу search, в том же порядке: отказы поиска переходят сюда как есть; пусто, если второй уровень не запускался."""
    timings: dict[str, float] = field(default_factory=dict)
    """Секунды на шаг."""

    @property
    def final(self) -> list[BottleCandidates | MatchedBottle | UnmatchedBottle]:
        """Ответ последнего отработавшего шага по каждой годной бутылке."""
        return list(self.resolution) if self.resolution else list(self.search)

    @property
    def bottles(self) -> list[BottleOutcome]:
        """Итог по каждой бутылке фото в порядке разметки: геометрия и ровно один исход после всех шагов."""
        verdicts = {v.crop.uuid: v for v in self.final}
        out = []
        for item in self.normalization:
            if isinstance(item, RejectedBottle):
                out.append(
                    BottleOutcome(
                        item.uuid,
                        item.score,
                        item.bottle,
                        item.label,
                        None,
                        None,
                        [],
                        item.rejection,
                    )
                )
                continue
            verdict = verdicts.get(item.uuid)
            match = verdict if isinstance(verdict, MatchedBottle) else None
            candidates = (
                verdict.candidates if isinstance(verdict, BottleCandidates) else []
            )
            rejection = (
                verdict.rejection if isinstance(verdict, UnmatchedBottle) else None
            )
            out.append(
                BottleOutcome(
                    item.uuid,
                    item.score,
                    item.bottle,
                    item.label,
                    item,
                    match,
                    candidates,
                    rejection,
                )
            )
        return out

    @property
    def items(self) -> list[BottleCrop | RejectedBottle]:
        """Итоговая разметка: бутылка, отвергнутая поиском или вторым уровнем, стоит здесь RejectedBottle с тем же uuid и причиной шага."""
        return replace_rejected(
            self.normalization,
            [
                RejectedBottle.of(r.crop, r.rejection)
                for r in self.final
                if isinstance(r, UnmatchedBottle)
            ],
        )

    @property
    def crops(self) -> list[BottleCrop]:
        """Бутылки, оставшиеся годными после всех шагов."""
        return [item for item in self.items if isinstance(item, BottleCrop)]

    @property
    def matched(self) -> list[MatchedBottle]:
        """Бутылки с выбранной позицией каталога, в порядке разметки."""
        return [r for r in self.resolution if isinstance(r, MatchedBottle)]


class Pipeline:
    """Оркестратор: нормализация → визуальный поиск → выбор позиции либо отказ.

    Pipeline() поднимает Normalizer(), VisSearcher() и второй уровень: FilterResolver для top_n, NearDuplicateResolver для groups, с карточками позиций
    из каталога, сам каталог — по config.yaml рядом с модулем (CSV локально либо в S3). Свой экземпляр шага или каталога передаётся
    аргументом. search=False — только нормализация, resolve=False — нормализация и поиск без второго уровня. С поиском при создании
    сверяется нормализатор: часть его конфига, определяющая пиксели кропов (Normalizer.crop_config), должна совпадать с той, что
    собирала каталог поиска, иначе ошибка с перечнем расхождений; каждая позиция коллекции поиска должна быть в каталоге, иначе ошибка
    с примерами — строка каталога приходит в ответ на любого кандидата. Второй уровень получает только бутылки с кандидатами, его ответы
    встают на их места.
    """

    def __init__(
        self,
        normalizer: Normalizer | None = None,
        searcher: Searcher | None = None,
        resolver: Resolver | None = None,
        catalog: Catalog | None = None,
        *,
        search: bool = True,
        resolve: bool = True,
    ) -> None:
        self.normalizer = normalizer or Normalizer()
        if catalog is None:
            cfg = load_config()
            catalog = load_catalog(cfg.catalog, cfg.slug_column)
        self.catalog = catalog
        self.searcher: Searcher | None = (searcher or VisSearcher()) if search else None
        self.resolver: Resolver | None = None
        if search and resolve:
            if resolver is not None:
                self.resolver = resolver
            elif isinstance(self.searcher, VisSearcher) and isinstance(self.searcher.cfg.search, TopNSearch):
                self.resolver = FilterResolver(cards=filter_cards_from_rows(catalog.rows, catalog.description))
            else:
                self.resolver = NearDuplicateResolver(cards=cards_from_rows(catalog.rows, catalog.description))
        if self.searcher is not None:
            self.searcher.check_normalization(self.normalizer.crop_config)
            missing = sorted(self.searcher.slugs() - catalog.rows.keys())
            if missing:
                raise RuntimeError(
                    f"в каталоге ({catalog.description}) нет {len(missing)} позиций коллекции поиска, например {missing[:5]}: "
                    "коллекция и каталог из разных выгрузок"
                )

    def __call__(self, img: Image.Image | Path | str) -> PipelineResult:
        """Картинка → выходы шагов: разметка нормализации, ответы поиска по годным бутылкам, ответы второго уровня, время каждого шага."""
        started = time.perf_counter()
        normalization = self.normalizer(img)
        result = PipelineResult(
            normalization, timings={"normalization": time.perf_counter() - started}
        )
        crops = [item for item in normalization if isinstance(item, BottleCrop)]
        if not crops or self.searcher is None:
            return result
        started = time.perf_counter()
        result.search = self.searcher(crops)
        result.timings["search"] = time.perf_counter() - started
        if self.resolver is None:
            return result
        started = time.perf_counter()
        found = [r for r in result.search if isinstance(r, BottleCandidates)]
        decided = iter(self.resolver(found))
        result.resolution = [
            next(decided) if isinstance(r, BottleCandidates) else r
            for r in result.search
        ]
        result.timings["resolve"] = time.perf_counter() - started
        return result
