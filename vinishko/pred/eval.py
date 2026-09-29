"""Ручка для скрипта оценки организаторов (test/participant_test.sh): POST /v1/eval/predict, фото одной бутылки → {"slug": ...}.

На фото ровно одна бутылка, поэтому из годных бутылок нормализации берётся бутылка с наибольшей площадью маски: на фото организаторов
цель стоит в центре и крупнее соседей по полке, которых режет край кадра. Скор отбора для этого не годится — у всех хорошо видимых
бутылок он около 0.98: на примере организаторов 02eef911 обрезанный сосед справа получил 0.980, а цель в центре 0.977 при площади
в 1.5 раза больше, и ответ был про соседа. На тесте hack-vine площадь и скор расходятся на 11 фото из 176, top-1 поиска без второго
уровня там верен в 4 и 5 случаях, но площадь дважды даёт верное семейство, которое дочитывает второй уровень. Нужна годная бутылка
с кропом, поэтому selection.max_bottles (MAX_BOTTLES) должен быть больше 1. Поиск и второй уровень идут только по выбранной бутылке.

Скрипт ждёт ответ не дольше 10 с, поэтому у ответа бюджет BUDGET_SECONDS от начала обработки: второй уровень, не уложившийся в него
или упавший, заменяется top-1 визуального поиска — он верен примерно в 90% случаев, а пустой ответ не верен никогда. Годной бутылки нет,
поиск или второй уровень отказали — {"slug": null}, скрипт запишет null.
"""

import asyncio
import threading
import time
from typing import Annotated

from fastapi import APIRouter
from fastapi import File
from fastapi import HTTPException
from fastapi import Request
from fastapi import UploadFile
from kostyl.utils import setup_logger
from pydantic import BaseModel
from pydantic import Field

from vinishko.pred.pipeline.pipeline import Resolver
from vinishko.pred.pipeline.steps.normalization.normalize import polys_area
from vinishko.pred.pipeline.structs import BottleCandidates
from vinishko.pred.pipeline.structs import BottleCrop
from vinishko.pred.pipeline.structs import MatchedBottle
from vinishko.pred.router import current
from vinishko.pred.router import read_picture


BUDGET_SECONDS = 8.0
"""Из 10 с скрипта: остальное — на загрузку фото и ответ."""

logger = setup_logger(fmt="detailed")
router = APIRouter(tags=["eval"])


class EvalResponse(BaseModel):
    """Ответ в формате скрипта организаторов."""

    slug: str | None = Field(
        description="Top-1 позиция каталога; null — годной бутылки нет либо её вина нет в каталоге"
    )


def search_top1(found: BottleCandidates) -> str:
    """Лучший кандидат визуального поиска."""
    return max(found.candidates, key=lambda c: c.score).slug


async def resolve(
    resolver: Resolver, found: BottleCandidates, remaining: float
) -> tuple[str | None, str]:
    """Второй уровень в пределах remaining секунд: slug и откуда он; не уложился или упал — top-1 поиска."""
    try:
        verdict = (
            await asyncio.wait_for(
                asyncio.to_thread(resolver, [found]), max(remaining, 0.0)
            )
        )[0]
    except TimeoutError:
        return (
            search_top1(found),
            f"поиск: второй уровень не уложился в {BUDGET_SECONDS:.0f} с",
        )
    except Exception as error:
        return (
            search_top1(found),
            f"поиск: второй уровень упал ({type(error).__name__}: {error})",
        )
    if isinstance(verdict, MatchedBottle):
        return verdict.candidate.slug, f"второй уровень, {verdict.source}"
    return None, f"второй уровень: {verdict.rejection.message}"


@router.post("/v1/eval/predict", response_model=EvalResponse)
async def predict(
    request: Request,
    image: Annotated[
        UploadFile, File(description="Фото одной бутылки: jpeg, png, webp, heic")
    ],
) -> EvalResponse:
    """Фото → slug бутылки с наибольшей площадью маски либо null."""
    started = time.perf_counter()
    picture = await read_picture(image)
    pipe = current(request)
    searcher = pipe.searcher
    if searcher is None:
        raise HTTPException(status_code=503, detail="поиск выключен")
    lock: threading.Lock = request.app.state.lock

    def locate() -> tuple[BottleCandidates | None, str]:
        """Нормализация и поиск под замком пайплайна: кандидаты выбранной бутылки либо почему их нет."""
        with lock:
            items = pipe.normalizer(picture)
            crops = [i for i in items if isinstance(i, BottleCrop)]
            if not crops:
                return None, f"годных бутылок нет, найдено {len(items)}"
            crop = max(crops, key=lambda c: polys_area(c.bottle))
            found = searcher([crop])[0]
        if isinstance(found, BottleCandidates):
            return found, ""
        return None, f"поиск: {found.rejection.message}"

    found, source = await asyncio.to_thread(locate)
    slug = None
    resolver = pipe.resolver
    if found is not None and resolver is None:
        slug, source = search_top1(found), "поиск, второй уровень выключен"
    elif found is not None and resolver is not None:
        remaining = BUDGET_SECONDS - (time.perf_counter() - started)
        slug, source = await resolve(resolver, found, remaining)
    elapsed = time.perf_counter() - started
    log = logger.warning if source.startswith("поиск: второй уровень") else logger.info
    log(f"eval {image.filename}: {slug or 'null'} — {source}, {elapsed:.2f} с")
    return EvalResponse(slug=slug)
