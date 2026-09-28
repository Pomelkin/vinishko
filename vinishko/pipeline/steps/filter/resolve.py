"""Фильтр: выбор одной позиции среди всех кандидатов top_n либо отказ. Адаптер к кропам в памяти."""

import copy
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import BoundedSemaphore

import numpy as np
from PIL import Image

from vinishko.pipeline.steps.vis_searcher.storage import encode_image
from vinishko.pipeline.structs import BottleCandidates
from vinishko.pipeline.structs import Candidate
from vinishko.pipeline.structs import MatchedBottle
from vinishko.pipeline.structs import Reason
from vinishko.pipeline.structs import Rejection
from vinishko.pipeline.structs import UnmatchedBottle

from .configs import NdrSettings
from .configs import load_config
from .models import NOT_FOUND
from .predictor import load_openrouter_env
from .predictor import missing_api_key_message
from .predictor import predict


HERE = Path(__file__).resolve().parent
NDR_CONCURRENCY = 32
NDR_SEMAPHORE = BoundedSemaphore(NDR_CONCURRENCY)
NDR_EXECUTOR = ThreadPoolExecutor(max_workers=NDR_CONCURRENCY)
QUERY_MAX_SIDE = 1600
"""Кроп запроса крупнее этой стороны уменьшается перед отправкой модели."""
SOURCE_MODEL = "filter_v1"
"""MatchedBottle.source: модель сравнила всех кандидатов поиска."""
PROMPT_FILES = (
    "resolve_multiple_same",
    "compare_year_matters",
    "compare_year_not_matter",
)
CSV_FIELDS = {
    "name": "Название вина",
    "winery": "Винодельня",
    "vintage": "Винтаж",
    "grapes": "Сорт винограда",
    "category": "Категория",
    "sugar": "Сахар",
    "sparkling": "Игристое",
    "abv": "Крепость, %",
    "aging_or_reserve": "Выдержка или резерв",
}
"""Колонки каталога для карточки позиции, если карточки берутся из локального CSV, а не из метаданных точки."""


class FilterReason(Reason):
    """Почему фильтр не выбрал ни одну позицию."""

    __stage__ = "resolve"
    NOT_FOUND = (
        "filter_not_found",
        "Точного совпадения нет",
        "Среди близких позиций каталога модель не нашла точный товар.",
    )


class FilterError(RuntimeError):
    """Модель или кандидаты не позволили получить проверенный ответ; top-1 вместо ответа не подставляется."""


def cards_from_rows(
    rows: dict[str, dict[str, str]], source: str = "каталог"
) -> dict[str, dict]:
    """Карточки позиций из строк CSV по slug; группы не требуются."""
    required = set(CSV_FIELDS.values())
    cards = {}
    for slug, row in rows.items():
        missing = required - set(row)
        if missing:
            raise FilterError(f"в {source} нет колонок {sorted(missing)}")
        cards[slug] = {
            "slug": slug,
            **{key: (row[column] or "").strip() for key, column in CSV_FIELDS.items()},
        }
    return cards


def load_cards(path: Path) -> dict[str, dict]:
    """Карточки позиций из локального CSV каталога."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if "Slug" not in (reader.fieldnames or []):
            raise FilterError(f"в {path} нет колонки Slug")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            slug = row["Slug"].strip()
            if not slug or slug in rows:
                raise FilterError(
                    f"пустой или повторный Slug в {path}: {slug!r}"
                )
            rows[slug] = row
    return cards_from_rows(rows, str(path))


def candidate_card(candidate: Candidate, cards: dict[str, dict] | None = None) -> dict:
    """Карточка позиции для NDR v5: поля из локального CSV, если он задан, иначе из метаданных точки; картинка — box_crop позиции из коллекции."""
    payload = candidate.payload
    card = cards.get(candidate.slug) if cards is not None else None
    if cards is not None and card is None:
        raise FilterError(
            f"у {candidate.slug} нет карточки в локальном каталоге"
        )
    image_bytes = encode_image(candidate.image, "png", 95)
    if card is not None:
        return card | {"reference_image_bytes": image_bytes}
    return {
        "slug": candidate.slug,
        "name": payload.get("name", ""),
        "winery": payload.get("winery", ""),
        "vintage": payload.get("vintage", ""),
        "grapes": payload.get("grapes") or payload.get("grape", ""),
        "category": payload.get("category", ""),
        "sugar": payload.get("sugar", ""),
        "sparkling": payload.get("sparkling", ""),
        "abv": payload.get("abv", ""),
        "aging_or_reserve": payload.get("aging_or_reserve", ""),
        "reference_image_bytes": image_bytes,
    }


class FilterResolver:
    """Кандидаты поиска по бутылке → ровно одна позиция каталога (MatchedBottle) либо отказ (UnmatchedBottle).

    Сравнивает всех переданных кандидатов без групп, даже единственного: один вызов модели через OpenRouter.
    QUERY — box_crop бутылки, ELEMENT — картинка и карточка каждой позиции; ответ строго ограничен переданными slug
    либо not_found. Ошибка API или контракта — FilterError, top-1 вместо ответа не подставляется.
    Без settings — config.yaml рядом с модулем; trace_dir — куда писать полный ответ модели по uuid бутылки; cards — карточки позиций
    из каталога (cards_from_rows по строкам CSV, Pipeline берёт их из своего каталога) вместо метаданных точки.
    """

    def __init__(
        self,
        settings: NdrSettings | None = None,
        trace_dir: Path | None = None,
        cards: dict[str, dict] | None = None,
    ) -> None:
        self.settings = settings if settings is not None else load_config()
        self.trace_dir = trace_dir
        self.cards = cards
        self.prompts = {
            name: {
                "content": (HERE / "prompts" / f"{name}.txt").read_text(
                    encoding="utf-8"
                )
            }
            for name in PROMPT_FILES
        }

    def __call__(
        self, found: list[BottleCandidates]
    ) -> list[MatchedBottle | UnmatchedBottle]:
        """Ответ по каждой бутылке, в том же порядке; вызовы модели идут параллельно, не больше NDR_CONCURRENCY разом."""
        if len(found) < 2:
            return [self._select(result) for result in found]
        return list(NDR_EXECUTOR.map(self._select, found))

    def with_trace_dir(self, trace_dir: Path | None) -> "FilterResolver":
        """Тот же выбор с другой директорией trace: для параллельного прогона разных фото."""
        worker = copy.copy(self)
        worker.trace_dir = trace_dir
        return worker

    def _select(self, result: BottleCandidates) -> MatchedBottle | UnmatchedBottle:
        candidates = result.candidates
        by_slug = {candidate.slug: candidate for candidate in candidates}
        if len(by_slug) != len(candidates):
            raise FilterError("slug кандидатов должны быть уникальны")

        load_openrouter_env()
        settings = self.settings
        provider = settings.openrouter
        if not (os.environ.get(provider.api_key_env) or "").strip():
            raise FilterError(missing_api_key_message(provider.api_key_env))
        generation = settings.generation
        runtime = {
            "model": os.environ.get("OPENROUTER_MODEL") or provider.model,
            "api_base": provider.api_base,
            "api_key_env": provider.api_key_env,
            "http_referer_env": provider.http_referer_env,
            "app_title_env": provider.app_title_env,
            "user_agent": provider.user_agent,
            "generation": generation.request_payload(),
            "extra_body": provider.extra_body,
            "image_detail": generation.image_detail,
            "provider": provider.routing.request_payload(),
            "timeout": settings.execution.timeout_seconds,
        }
        query_image = result.crop.box_crop
        if max(query_image.shape[:2]) > QUERY_MAX_SIDE:
            resized = Image.fromarray(query_image)
            resized.thumbnail(
                (QUERY_MAX_SIDE, QUERY_MAX_SIDE), Image.Resampling.LANCZOS
            )
            query_image = np.asarray(resized)
        request = {
            "query": {"image_bytes": encode_image(query_image, "png", 95)},
            "candidates": [candidate_card(candidate, self.cards) for candidate in candidates],
            "prompts": self.prompts,
            "runtime": runtime,
        }
        with NDR_SEMAPHORE:
            response = predict(request)
        self._write_trace(result.crop.uuid, response)
        if response["_status"] != "ok":
            raise FilterError(
                f"filter {response['_status']} для кандидатов {list(by_slug)}: {response['_error']}"
            )
        slug = response["slug"]
        checklist = response["_trace"]["comparison_calls"][0]["selected"]["checklist"]
        if slug == NOT_FOUND:
            observations = "; ".join(
                f"{key}: {value.get('observation', value) if isinstance(value, dict) else value}"
                for key, value in checklist.items()
            )
            detail = f"ни один из {len(candidates)} кандидатов не подошёл; {observations}"
            return UnmatchedBottle(
                result.crop, Rejection(FilterReason.NOT_FOUND, detail)
            )
        if slug not in by_slug:
            raise FilterError(f"модель вернула slug {slug!r}, которого нет среди кандидатов")
        return MatchedBottle(result.crop, by_slug[slug], SOURCE_MODEL, checklist)

    def _write_trace(self, uuid: str, response: dict) -> None:
        if self.trace_dir is None:
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        (self.trace_dir / f"{uuid}.json").write_text(
            json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
        )
