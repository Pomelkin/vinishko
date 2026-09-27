"""Второй уровень: выбор одной позиции каталога внутри группы near-duplicates. Адаптер NDR v5 к кандидатам поиска и кропам в памяти."""

import copy
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import BoundedSemaphore

import numpy as np
from PIL import Image

from vinishko.pipeline.steps.vis_searcher.catalog import FIELD_GROUP_SLUGS
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
SOURCE_VECTOR, SOURCE_MODEL = "vector", "ndr_v5"
"""MatchedBottle.source: в группе одна позиция, взята без модели; модель выбрала среди позиций группы."""
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


class NearDuplicateReason(Reason):
    """Почему второй уровень не выбрал ни одну позицию группы."""

    __stage__ = "resolve"
    NOT_FOUND = (
        "near_duplicate_not_found",
        "Точного совпадения нет",
        "Среди близких позиций каталога модель не нашла точный товар.",
    )


class NearDuplicateError(RuntimeError):
    """Модель или данные группы не позволили получить проверенный ответ; top-1 вместо ответа не подставляется."""


def cards_from_rows(
    rows: dict[str, dict[str, str]], source: str = "каталог"
) -> dict[str, dict]:
    """Карточки позиций для модели из строк CSV каталога по slug: поля CSV_FIELDS, имя фото и группа."""
    required = {"image_filename", "near_duplicate_group_slug", *CSV_FIELDS.values()}
    cards = {}
    for slug, row in rows.items():
        missing = required - set(row)
        if missing:
            raise NearDuplicateError(f"в {source} нет колонок {sorted(missing)}")
        cards[slug] = {
            "slug": slug,
            "image_filename": row["image_filename"].strip(),
            "group": row["near_duplicate_group_slug"].strip() or slug,
            **{key: (row[column] or "").strip() for key, column in CSV_FIELDS.items()},
        }
    return cards


def load_cards(path: Path) -> dict[str, dict]:
    """Карточки позиций из локального CSV каталога."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if "Slug" not in (reader.fieldnames or []):
            raise NearDuplicateError(f"в {path} нет колонки Slug")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            slug = row["Slug"].strip()
            if not slug or slug in rows:
                raise NearDuplicateError(
                    f"пустой или повторный Slug в {path}: {slug!r}"
                )
            rows[slug] = row
    return cards_from_rows(rows, str(path))


def candidate_card(candidate: Candidate, cards: dict[str, dict] | None = None) -> dict:
    """Карточка позиции для NDR v5: поля из локального CSV, если он задан, иначе из метаданных точки; картинка — box_crop позиции из коллекции."""
    payload = candidate.payload
    card = cards.get(candidate.slug) if cards is not None else None
    if cards is not None and card is None:
        raise NearDuplicateError(
            f"у {candidate.slug} нет карточки в локальном каталоге"
        )
    if card is not None and card["group"] != candidate.group:
        raise NearDuplicateError(
            f"группа {candidate.slug} в локальном каталоге и в коллекции не совпадает"
        )
    image_bytes = encode_image(candidate.image, "png", 95)
    if card is not None:
        return {
            key: value
            for key, value in card.items()
            if key not in {"image_filename", "group"}
        } | {"reference_image_bytes": image_bytes}
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


class NearDuplicateResolver:
    """Кандидаты поиска по бутылке → ровно одна позиция каталога (MatchedBottle) либо отказ (UnmatchedBottle).

    Работает с группой top-1 кандидата и требует её целиком, поэтому поиск должен идти в режиме search.mode: groups: в top_n члены
    группы за пределами top_k в кандидаты не попадают. Группа из одной позиции — ответ без модели. Из нескольких — один вызов модели
    через OpenRouter: QUERY — box_crop бутылки, ELEMENT — картинка и карточка каждой позиции группы; ответ строго ограничен slug группы
    либо not_found. Ошибка API или контракта — NearDuplicateError, top-1 вместо ответа не подставляется.
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

    def with_trace_dir(self, trace_dir: Path | None) -> "NearDuplicateResolver":
        """Тот же выбор с другой директорией trace: для параллельного прогона разных фото."""
        worker = copy.copy(self)
        worker.trace_dir = trace_dir
        return worker

    def _select(self, result: BottleCandidates) -> MatchedBottle | UnmatchedBottle:
        top = result.candidates[0]
        members = top.payload.get(FIELD_GROUP_SLUGS)
        if (
            not isinstance(members, list)
            or not members
            or len(set(members)) != len(members)
        ):
            raise NearDuplicateError(
                f"у {top.slug} некорректный group_slugs в коллекции"
            )
        candidates = [
            candidate for candidate in result.candidates if candidate.group == top.group
        ]
        by_slug = {candidate.slug: candidate for candidate in candidates}
        if set(by_slug) != set(members):
            raise NearDuplicateError(
                f"группа {top.group} в кандидатах неполна: ожидались {members}, получены {list(by_slug)}; второй уровень требует search.mode: groups"
            )
        if len(members) == 1:
            return MatchedBottle(result.crop, top, SOURCE_VECTOR, {})

        load_openrouter_env()
        settings = self.settings
        provider = settings.openrouter
        if not (os.environ.get(provider.api_key_env) or "").strip():
            raise NearDuplicateError(missing_api_key_message(provider.api_key_env))
        generation = settings.generation
        runtime = {
            "model": os.environ.get("OPENROUTER_MODEL") or provider.model,
            "api_base": provider.api_base,
            "api_key_env": provider.api_key_env,
            "http_referer_env": provider.http_referer_env,
            "app_title_env": provider.app_title_env,
            "user_agent": provider.user_agent,
            "generation": generation.request_payload(),
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
            "group": {
                "candidates": [
                    candidate_card(candidate, self.cards) for candidate in candidates
                ]
            },
            "prompts": self.prompts,
            "runtime": runtime,
        }
        with NDR_SEMAPHORE:
            response = predict(request)
        self._write_trace(result.crop.uuid, response)
        if response["_status"] != "ok":
            raise NearDuplicateError(
                f"NDR {response['_status']} для группы {top.group}: {response['_error']}"
            )
        slug = response["slug"]
        checklist = response["_trace"]["comparison_calls"][0]["selected"]["checklist"]
        if slug == NOT_FOUND:
            observations = "; ".join(
                f"{key}: {value.get('observation', value) if isinstance(value, dict) else value}"
                for key, value in checklist.items()
            )
            detail = f"группа {top.group}: ни один из {len(members)} кандидатов не подошёл; {observations}"
            return UnmatchedBottle(
                result.crop, Rejection(NearDuplicateReason.NOT_FOUND, detail)
            )
        return MatchedBottle(result.crop, by_slug[slug], SOURCE_MODEL, checklist)

    def _write_trace(self, uuid: str, response: dict) -> None:
        if self.trace_dir is None:
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        (self.trace_dir / f"{uuid}.json").write_text(
            json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
        )
