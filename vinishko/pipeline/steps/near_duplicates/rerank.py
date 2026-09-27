"""Адаптер NDR v5: группа из Qdrant и кропы в памяти → точный slug или отказ."""

import csv
import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import BoundedSemaphore

import numpy as np
from PIL import Image

from vinishko.pipeline.steps.vis_searcher.catalog import FIELD_GROUP_SLUGS
from vinishko.pipeline.steps.vis_searcher.configs import VisSearcherConfig
from vinishko.pipeline.steps.vis_searcher.storage import ImageStore, encode_image, make_store
from vinishko.pipeline.structs import BottleCandidates, Candidate, Reason, UnmatchedBottle

from .config import SETTINGS, NdrSettings
from .models import NOT_FOUND
from .predictor import predict

HERE = Path(__file__).resolve().parent
NDR_CONCURRENCY = 32
NDR_SEMAPHORE = BoundedSemaphore(NDR_CONCURRENCY)
NDR_EXECUTOR = ThreadPoolExecutor(max_workers=NDR_CONCURRENCY)
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


class NearDuplicateReason(Reason):
    """Почему NDR не выбрал ни один SKU группы."""

    NOT_FOUND = (
        "near_duplicate_not_found",
        "Точного совпадения нет",
        "Среди близких позиций каталога модель не нашла точный товар.",
    )


class NearDuplicateError(RuntimeError):
    """Модель или данные группы не позволили выбрать проверенный результат."""


def load_cards(path: Path) -> dict[str, dict]:
    """Читает авторитетные карточки и имена Эталонов из локального CSV."""
    cards = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"Slug", "image_filename", "near_duplicate_group_slug", *CSV_FIELDS.values()}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise NearDuplicateError(f"в {path} нет колонок {sorted(missing)}")
        for row in reader:
            slug = row["Slug"].strip()
            if not slug or slug in cards:
                raise NearDuplicateError(f"пустой или повторный Slug в {path}: {slug!r}")
            cards[slug] = {
                "slug": slug,
                "image_filename": row["image_filename"].strip(),
                "group": row["near_duplicate_group_slug"].strip() or slug,
                **{key: (row[column] or "").strip() for key, column in CSV_FIELDS.items()},
            }
    return cards


def candidate_card(candidate: Candidate, reference_store: ImageStore | None = None, cards: dict[str, dict] | None = None) -> dict:
    """Поля карточки Qdrant в формате NDR v5."""
    payload = candidate.payload
    card = cards.get(candidate.slug) if cards is not None else None
    if cards is not None and card is None:
        raise NearDuplicateError(f"у {candidate.slug} нет карточки в локальном Каталоге")
    if card is not None and card["group"] != candidate.group:
        raise NearDuplicateError(f"группа {candidate.slug} в локальном Каталоге и Qdrant не совпадает")
    if payload.get("image_crop") == "bottle_box" or reference_store is None:
        image_bytes = encode_image(candidate.image, "png", 95)
    else:
        source_image = card["image_filename"] if card is not None else payload.get("source_image")
        if not source_image:
            raise NearDuplicateError(f"у {candidate.slug} нет source_image для NDR")
        image_bytes = reference_store.get_bytes(source_image)
    if card is not None:
        return {key: value for key, value in card.items() if key not in {"image_filename", "group"}} | {"reference_image_bytes": image_bytes}
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


class NearDuplicateReranker:
    """Берёт группу первого векторного кандидата и вызывает NDR v5 только для группы из 2+ SKU."""

    def __init__(self, settings: NdrSettings = SETTINGS, trace_dir: Path | None = None, reference_store: ImageStore | None = None, catalog_csv: Path | None = None) -> None:
        self.settings = settings
        self.trace_dir = trace_dir
        self.reference_store = reference_store
        self.cards = load_cards(catalog_csv) if catalog_csv is not None else None
        self.prompts = {
            name: {"content": (HERE / "prompts" / f"{name}.txt").read_text(encoding="utf-8")}
            for name in PROMPT_FILES
        }

    @classmethod
    def from_search_config(cls, cfg: VisSearcherConfig | None, trace_dir: Path | None = None) -> "NearDuplicateReranker":
        """Создаёт хранилище исходных эталонов из конфига поиска."""
        store = make_store(cfg.reference_images, cfg.cache_dir) if cfg is not None and cfg.reference_images is not None else None
        return cls(trace_dir=trace_dir, reference_store=store, catalog_csv=cfg.catalog_csv if cfg is not None else None)

    def __call__(self, results: list[BottleCandidates | UnmatchedBottle]) -> list[BottleCandidates | UnmatchedBottle]:
        """Сохраняет порядок бутылок; каждый ответ содержит один SKU либо отказ."""
        if len(results) < 2:
            return [self._select(result) if isinstance(result, BottleCandidates) else result for result in results]
        return list(NDR_EXECUTOR.map(
            lambda result: self._select(result) if isinstance(result, BottleCandidates) else result,
            results,
        ))

    def with_trace_dir(self, trace_dir: Path | None) -> "NearDuplicateReranker":
        """Отдельный путь trace для параллельного прогона разных фото."""
        worker = copy.copy(self)
        worker.trace_dir = trace_dir
        return worker

    def _select(self, result: BottleCandidates) -> BottleCandidates | UnmatchedBottle:
        top = result.candidates[0]
        members = top.payload.get(FIELD_GROUP_SLUGS)
        if not isinstance(members, list) or not members or len(set(members)) != len(members):
            raise NearDuplicateError(f"у {top.slug} некорректный group_slugs в Qdrant")
        candidates = [candidate for candidate in result.candidates if candidate.group == top.group]
        by_slug = {candidate.slug: candidate for candidate in candidates}
        if set(by_slug) != set(members):
            raise NearDuplicateError(f"группа {top.group} неполна в выдаче: ожидались {members}, получены {list(by_slug)}")
        if len(members) == 1:
            return BottleCandidates(result.crop, [top], selection={"source": "vector", "slug": top.slug})

        settings = self.settings
        provider = settings.openrouter
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
        query_image = result.crop.original if result.crop.original is not None else result.crop.box_crop
        if max(query_image.shape[:2]) > 1600:
            resized = Image.fromarray(query_image)
            resized.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            query_image = np.asarray(resized)
        request = {
            "query": {"image_bytes": encode_image(query_image, "png", 95)},
            "group": {"candidates": [candidate_card(candidate, self.reference_store, self.cards) for candidate in candidates]},
            "prompts": self.prompts,
            "runtime": runtime,
        }
        with NDR_SEMAPHORE:
            response = predict(request)
        self._write_trace(result.crop.uuid, response)
        if response["_status"] != "ok":
            raise NearDuplicateError(f"NDR {response['_status']} для группы {top.group}: {response['_error']}")
        slug = response["slug"]
        selected = response["_trace"]["comparison_calls"][0]["selected"]
        selection = {"source": "ndr_v5", "slug": slug, "checklist": selected["checklist"]}
        if slug == NOT_FOUND:
            detail = f"группа {top.group}: ни один из {len(members)} кандидатов не подошёл"
            return UnmatchedBottle(result.crop, result.crop.reject(NearDuplicateReason.NOT_FOUND, detail), selection=selection)
        return BottleCandidates(result.crop, [by_slug[slug]], selection=selection)

    def _write_trace(self, uuid: str, response: dict) -> None:
        if self.trace_dir is None:
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        (self.trace_dir / f"{uuid}.json").write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
