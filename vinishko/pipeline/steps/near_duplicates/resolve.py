"""Второй уровень: выбор одной позиции каталога среди кандидатов поиска — внутри каждой группы near-duplicates, затем среди лучших
позиций групп. Адаптер NDR v5 к кандидатам поиска и кропам в памяти."""

import copy
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import BoundedSemaphore

import numpy as np
from PIL import Image

from vinishko.pipeline.steps.vis_searcher.catalog import FIELD_GROUP_SLUGS
from vinishko.pipeline.steps.vis_searcher.storage import encode_image
from vinishko.pipeline.structs import BottleCandidates
from vinishko.pipeline.structs import BottleCrop
from vinishko.pipeline.structs import Candidate
from vinishko.pipeline.structs import MatchedBottle
from vinishko.pipeline.structs import Reason
from vinishko.pipeline.structs import Rejection
from vinishko.pipeline.structs import UnmatchedBottle

from .configs import NdrSettings
from .configs import load_config
from .models import NOT_FOUND
from .predictor import TASK_FINAL
from .predictor import TASK_GROUP
from .predictor import TASK_PROMPTS
from .predictor import Task
from .predictor import load_openrouter_env
from .predictor import missing_api_key_message
from .predictor import predict


HERE = Path(__file__).resolve().parent
QUERY_MAX_SIDE = 1600
"""Кроп запроса крупнее этой стороны уменьшается перед отправкой модели."""
IMAGE_QUALITY = 92
"""JPEG-качество картинок для модели, запроса и позиций: в 4 раза легче PNG при том же разрешении и числе токенов, надписи читаются.
Замер 2026-09-28 на 60 картинках: медиана 303 КБ в PNG против 73 КБ, тело запроса в среднем 2.8 МБ вместо ~0.7."""
SOURCE_GROUP, SOURCE_FINAL = TASK_GROUP, TASK_FINAL
"""MatchedBottle.source — вызов, который выбрал позицию: group — модель выбрала её внутри группы, других финалистов не было;
final — модель выбрала её среди финалистов групп либо подтвердила единственную позицию-одиночку."""
PROMPT_FILES = (
    *TASK_PROMPTS.values(),
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
    """Почему второй уровень не выбрал ни одну позицию."""

    __stage__ = "resolve"
    NOT_FOUND = (
        "near_duplicate_not_found",
        "Точного совпадения нет",
        "Модель сравнила кандидатов поиска — позиции внутри каждой группы и лучшие позиции групп между собой — и не нашла среди них этот товар.",
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
    image_bytes = encode_image(candidate.image, "jpg", IMAGE_QUALITY)
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


@dataclass(frozen=True, slots=True)
class Call:
    """Один вызов модели: бутылка, задача, позиции в порядке поиска и имя файла trace."""

    crop: BottleCrop
    query: bytes
    """box_crop бутылки в JPEG."""
    task: Task
    candidates: list[Candidate]
    trace_name: str
    """group<N> — N-я группа кандидатов бутылки; final."""


@dataclass(frozen=True, slots=True)
class Pick:
    """Ответ одного вызова модели."""

    candidate: Candidate | None
    """Выбранная позиция; None — not_found."""
    checklist: dict
    """Наблюдения модели по производителю, профилю и году."""


def split_groups(result: BottleCandidates) -> list[list[Candidate]]:
    """Кандидаты по группам в порядке поиска; группа сверяется с group_slugs точки, неполная — ошибка."""
    groups: dict[str, list[Candidate]] = {}
    for candidate in result.candidates:
        groups.setdefault(candidate.group, []).append(candidate)
    for name, members in groups.items():
        expected = members[0].payload.get(FIELD_GROUP_SLUGS)
        if (
            not isinstance(expected, list)
            or not expected
            or len(set(expected)) != len(expected)
        ):
            raise NearDuplicateError(
                f"у {members[0].slug} некорректный group_slugs в коллекции"
            )
        got = [candidate.slug for candidate in members]
        if len(got) != len(expected) or set(got) != set(expected):
            raise NearDuplicateError(
                f"группа {name} в кандидатах неполна: ожидались {expected}, получены {got}; второй уровень требует search.mode: groups"
            )
    return list(groups.values())


def query_jpeg(crop: BottleCrop) -> bytes:
    """box_crop бутылки для модели: не крупнее QUERY_MAX_SIDE по большей стороне, JPEG."""
    image = crop.box_crop
    if max(image.shape[:2]) > QUERY_MAX_SIDE:
        resized = Image.fromarray(image)
        resized.thumbnail((QUERY_MAX_SIDE, QUERY_MAX_SIDE), Image.Resampling.LANCZOS)
        image = np.asarray(resized)
    return encode_image(image, "jpg", IMAGE_QUALITY)


def observations(checklist: dict) -> str:
    """Наблюдения модели одной строкой."""
    return "; ".join(
        f"{key}: {value.get('observation', value) if isinstance(value, dict) else value}"
        for key, value in checklist.items()
    )


class NearDuplicateResolver:
    """Кандидаты поиска по бутылке → ровно одна позиция каталога (MatchedBottle) либо отказ (UnmatchedBottle), в два круга.

    Кандидаты делятся на группы одинакового дизайна в порядке поиска. Каждая группа нужна целиком, поэтому поиск должен идти в режиме
    search.mode: groups: в top_n члены группы за пределами top_k в кандидаты не попадают, и шаг падает с NearDuplicateError. Сколько групп
    приходит, решает поиск: top_groups, group_threshold и group_margin его конфига.
    Круг групп: по каждой группе из нескольких позиций один вызов модели — лучшая позиция группы либо not_found; позиция без группы
    (группа из одной) проходит дальше без вызова. Финал: финалисты — выбранные в группах и одиночки, в порядке поиска; один вызов модели
    выбирает среди них позицию либо not_found, с одним финалистом-одиночкой это проверка. Финал не нужен, если финалист один и его уже
    выбрала модель в своей группе; финалистов нет — отказ. Вызовы одного круга идут параллельно по всем бутылкам, не больше
    execution.concurrency разом на резолвер: лимит общий для всех вызовов экземпляра и его копий with_trace_dir, в том числе из разных потоков. QUERY — box_crop бутылки, ELEMENT — картинка и карточка позиции; ответ строго ограничен slug позиций вызова либо not_found.
    Ошибка API или контракта — NearDuplicateError, top-1 вместо ответа не подставляется.
    Без settings — config.yaml рядом с модулем; trace_dir — куда писать полный ответ модели на каждый вызов: <uuid бутылки>/group<N>.json
    и final.json; cards — карточки позиций из каталога (cards_from_rows по строкам CSV, Pipeline берёт их из своего каталога) вместо
    метаданных точки.
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
        limit = self.settings.execution.concurrency
        self.semaphore = BoundedSemaphore(limit)
        self.executor = ThreadPoolExecutor(max_workers=limit)
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
        """Ответ по каждой бутылке, в том же порядке: круг групп по всем бутылкам, потом финал по всем бутылкам."""
        groups = [split_groups(result) for result in found]
        queries = [query_jpeg(result.crop) for result in found]
        group_calls = {
            (i, rank): Call(
                result.crop, queries[i], TASK_GROUP, members, f"group{rank}"
            )
            for i, result in enumerate(found)
            for rank, members in enumerate(groups[i], 1)
            if len(members) > 1
        }
        answers = self._run(list(group_calls.values()))
        picks = dict(zip(group_calls, answers, strict=True))
        verdicts: dict[int, MatchedBottle | UnmatchedBottle] = {}
        final_calls: dict[int, Call] = {}
        for i, result in enumerate(found):
            finalists: list[tuple[Candidate, Pick | None]] = []
            for rank, members in enumerate(groups[i], 1):
                if len(members) == 1:
                    finalists.append((members[0], None))
                elif (pick := picks[i, rank]).candidate is not None:
                    finalists.append((pick.candidate, pick))
            if not finalists:
                best = picks[i, 1]
                verdicts[i] = UnmatchedBottle(
                    result.crop,
                    Rejection(
                        NearDuplicateReason.NOT_FOUND,
                        f"модель отвергла все позиции в каждой из {len(groups[i])} групп; "
                        f"лучшая по поиску группа {groups[i][0][0].group}: {observations(best.checklist)}",
                    ),
                )
            elif len(finalists) == 1 and (pick := finalists[0][1]) is not None:
                verdicts[i] = MatchedBottle(
                    result.crop, finalists[0][0], SOURCE_GROUP, pick.checklist
                )
            else:
                final_calls[i] = Call(
                    result.crop,
                    queries[i],
                    TASK_FINAL,
                    [candidate for candidate, _ in finalists],
                    TASK_FINAL,
                )
        for (i, call), pick in zip(
            final_calls.items(), self._run(list(final_calls.values())), strict=True
        ):
            if pick.candidate is None:
                verdicts[i] = UnmatchedBottle(
                    call.crop,
                    Rejection(
                        NearDuplicateReason.NOT_FOUND,
                        f"модель отвергла финалистов {', '.join(c.slug for c in call.candidates)}: {observations(pick.checklist)}",
                    ),
                )
            else:
                verdicts[i] = MatchedBottle(
                    call.crop, pick.candidate, SOURCE_FINAL, pick.checklist
                )
        return [verdicts[i] for i in range(len(found))]

    def with_trace_dir(self, trace_dir: Path | None) -> "NearDuplicateResolver":
        """Тот же выбор с другой директорией trace: для параллельного прогона разных фото."""
        worker = copy.copy(self)
        worker.trace_dir = trace_dir
        return worker

    def _run(self, calls: list[Call]) -> list[Pick]:
        """Вызовы параллельно, ответы в том же порядке."""
        if len(calls) < 2:
            return [self._ask(call) for call in calls]
        return list(self.executor.map(self._ask, calls))

    def _ask(self, call: Call) -> Pick:
        """Один вызов модели; ответ вне контракта либо ошибка API — NearDuplicateError после записи trace."""
        load_openrouter_env()
        provider = self.settings.openrouter
        if not (os.environ.get(provider.api_key_env) or "").strip():
            raise NearDuplicateError(missing_api_key_message(provider.api_key_env))
        generation = self.settings.generation
        request = {
            "task": call.task,
            "query": {"image_bytes": call.query},
            "candidates": [
                candidate_card(candidate, self.cards) for candidate in call.candidates
            ],
            "prompts": self.prompts,
            "runtime": {
                "model": os.environ.get("OPENROUTER_MODEL") or provider.model,
                "api_base": provider.api_base,
                "api_key_env": provider.api_key_env,
                "http_referer_env": provider.http_referer_env,
                "app_title_env": provider.app_title_env,
                "user_agent": provider.user_agent,
                "generation": generation.request_payload(call.task),
                "image_detail": generation.image_detail,
                "provider": provider.routing.request_payload(),
                "timeout": self.settings.execution.timeout_seconds,
                "retries": self.settings.execution.retries,
            },
        }
        with self.semaphore:
            response = predict(request)
        self._write_trace(call, response)
        slugs = [candidate.slug for candidate in call.candidates]
        if response["_status"] != "ok":
            raise NearDuplicateError(
                f"NDR {response['_status']} на вызове {call.trace_name} по {slugs}: {response['_error']}"
            )
        checklist = response["_trace"]["comparison_calls"][0]["selected"]["checklist"]
        if response["slug"] == NOT_FOUND:
            return Pick(None, checklist)
        return Pick(call.candidates[slugs.index(response["slug"])], checklist)

    def _write_trace(self, call: Call, response: dict) -> None:
        if self.trace_dir is None:
            return
        path = self.trace_dir / call.crop.uuid / f"{call.trace_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "task": call.task,
                    "candidates": [candidate.slug for candidate in call.candidates],
                    **response,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
