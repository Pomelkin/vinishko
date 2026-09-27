"""Top-3 Qdrant → один SKU через VLM, без групповой логики."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import httpx2
from openai import OpenAI

from vinishko.pipeline.steps.vis_searcher.configs import VisSearcherConfig
from vinishko.pipeline.steps.vis_searcher.storage import ImageStore, encode_image, make_store
from vinishko.pipeline.structs import BottleCandidates, Candidate, Reason, UnmatchedBottle

from .config import VanillaVlmSettings
from .models import NOT_FOUND
from .predictor import classify

HERE = Path(__file__).resolve().parent
PROMPT_FILES = ("resolve_multiple_same", "compare_year_matters", "compare_year_not_matter")
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


class VanillaVlmReason(Reason):
    NOT_FOUND = (
        "vanilla_vlm_not_found",
        "Точного совпадения нет",
        "Среди ближайших позиций каталога модель не нашла точный товар.",
    )


class VanillaVlmError(RuntimeError):
    """Запрос к VLM или ответ модели не позволил выбрать SKU."""


def load_cards(path: Path) -> dict[str, dict[str, str]]:
    """Читать карточки каталога без зависимости от колонки групп."""
    cards: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"Slug", "image_filename", *CSV_FIELDS.values()}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise VanillaVlmError(f"в {path} нет колонок {sorted(missing)}")
        for row in reader:
            slug = row["Slug"].strip()
            if not slug or slug in cards:
                raise VanillaVlmError(f"пустой или повторный Slug в {path}: {slug!r}")
            cards[slug] = {
                "slug": slug,
                "image_filename": (row["image_filename"] or "").strip(),
                **{key: (row[column] or "").strip() for key, column in CSV_FIELDS.items()},
            }
    return cards


def candidate_card(candidate: Candidate, reference_store: ImageStore | None, cards: dict[str, dict[str, str]] | None) -> dict[str, Any]:
    """Собрать поля и изображение кандидата для мультимодального запроса."""
    card = cards.get(candidate.slug) if cards is not None else None
    if cards is not None and card is None:
        raise VanillaVlmError(f"у {candidate.slug} нет карточки в локальном Каталоге")
    if reference_store is None:
        image_bytes = encode_image(candidate.image, "png", 95)
    else:
        source_image = card["image_filename"] if card is not None else candidate.payload.get("source_image")
        if not source_image:
            raise VanillaVlmError(f"у {candidate.slug} нет source_image для VLM")
        image_bytes = reference_store.get_bytes(source_image)
    fields = card or {"slug": candidate.slug, **candidate.payload}
    return {
        "slug": candidate.slug,
        **{key: fields.get(key, "") for key in CSV_FIELDS},
        "reference_image_bytes": image_bytes,
    }


class VanillaVlmReranker:
    """Классифицировать top-3 без чтения group и group_slugs."""

    def __init__(
        self,
        trace_dir: Path | None = None,
        reference_store: ImageStore | None = None,
        catalog_csv: Path | None = None,
        client: OpenAI | None = None,
        settings: VanillaVlmSettings | None = None,
    ) -> None:
        self.trace_dir = trace_dir
        self.reference_store = reference_store
        self.cards = load_cards(catalog_csv) if catalog_csv is not None else None
        self.prompts = {name: (HERE / "prompts" / f"{name}.txt").read_text(encoding="utf-8") for name in PROMPT_FILES}
        self.client = client
        self.settings = settings or VanillaVlmSettings()

    @classmethod
    def from_search_config(cls, cfg: VisSearcherConfig | None, trace_dir: Path | None = None) -> VanillaVlmReranker:
        store = make_store(cfg.reference_images, cfg.cache_dir) if cfg is not None and cfg.reference_images is not None else None
        return cls(trace_dir=trace_dir, reference_store=store, catalog_csv=cfg.catalog_csv if cfg is not None else None)

    def _client(self) -> OpenAI:
        if self.client is None:
            if self.settings.api_key is None or not self.settings.api_key.get_secret_value():
                raise VanillaVlmError("OPENROUTER_API_KEY не задан")
            headers = {"User-Agent": "vinishko-vanilla-vlm/1"}
            if site_url := self.settings.site_url:
                headers["HTTP-Referer"] = site_url
            if title := self.settings.app_title:
                headers["X-Title"] = title
            self.client = OpenAI(
                api_key=self.settings.api_key.get_secret_value(),
                base_url=self.settings.api_base,
                timeout=self.settings.timeout_seconds,
                max_retries=0,
                default_headers=headers,
                http_client=httpx2.Client(proxy=self.settings.http_proxy or None, trust_env=False),
            )
        return self.client

    def __call__(self, results: list[BottleCandidates | UnmatchedBottle]) -> list[BottleCandidates | UnmatchedBottle]:
        return [self._select(result) if isinstance(result, BottleCandidates) else result for result in results]

    def _select(self, result: BottleCandidates) -> BottleCandidates | UnmatchedBottle:
        candidates = result.candidates
        by_slug = {candidate.slug: candidate for candidate in candidates}
        if len(by_slug) != len(candidates):
            raise VanillaVlmError("повторные slug в выдаче поиска")
        try:
            cards = [candidate_card(candidate, self.reference_store, self.cards) for candidate in candidates]
            selected, trace = classify(
                self._client(),
                self.settings,
                encode_image(result.crop.crop, "png", 95),
                cards,
                self.prompts,
            )
        except Exception as error:
            self._write_trace(result.crop.uuid, {"candidate_slugs": list(by_slug), "error": f"{type(error).__name__}: {error}"})
            raise VanillaVlmError(f"vanilla_vlm для кандидатов {list(by_slug)}: {error}") from error
        self._write_trace(result.crop.uuid, trace)
        slug = selected["slug"]
        selection = {"source": "vanilla_vlm", "slug": slug, "checklist": selected["checklist"]}
        if slug == NOT_FOUND:
            rejected = result.crop.reject(VanillaVlmReason.NOT_FOUND, "VLM исключила всех ближайших кандидатов")
            return UnmatchedBottle(result.crop, rejected, selection=selection)
        if slug not in by_slug:
            raise VanillaVlmError(f"VLM вернула slug вне top-3: {slug}")
        return BottleCandidates(result.crop, [by_slug[slug]], selection=selection)

    def _write_trace(self, uuid: str, trace: dict[str, Any]) -> None:
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            (self.trace_dir / f"{uuid}.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
