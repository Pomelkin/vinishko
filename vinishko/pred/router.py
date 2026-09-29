"""Ручки распознавания поверх пайплайна.

POST /recognize принимает фото и отдаёт список бутылок, дошедших до поиска: маска на исходном фото, скор отбора и исход — выбранная
позиция каталога (slug, скор, строка каталога, картинка из коллекции; то же у каждого кандидата, когда второй уровень выключен) либо отказ поиска или второго уровня с причиной. Объекты, отброшенные
нормализацией (не целевая бутылка, нет читаемой этикетки), в список не входят, только их число. GET /catalog/images/<имя> отдаёт картинку позиции,
GET /health — состояние. Пайплайн и замок кладёт в app.state приложение vinishko.app при старте; запросы к пайплайну идут по одному, под замком: SAM3 и engine
TensorRT не рассчитаны на параллельные вызовы из одного процесса.
"""

import asyncio
import os
import threading
import time
from io import BytesIO
from typing import Annotated
from typing import cast

from fastapi import APIRouter
from fastapi import File
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi import UploadFile
from PIL import Image
from PIL import UnidentifiedImageError

from vinishko.pred.pipeline.catalog import Catalog
from vinishko.pred.pipeline.pipeline import Pipeline
from vinishko.pred.pipeline.pipeline import PipelineResult
from vinishko.pred.pipeline.steps.near_duplicates import NearDuplicateResolver
from vinishko.pred.pipeline.steps.normalization.seg import open_image
from vinishko.pred.pipeline.steps.vis_searcher.catalog import FIELD_IMAGE
from vinishko.pred.pipeline.steps.vis_searcher.search import VisSearcher
from vinishko.pred.pipeline.steps.vis_searcher.storage import encode_image
from vinishko.pred.pipeline.structs import BottleOutcome
from vinishko.pred.pipeline.structs import Candidate
from vinishko.pred.pipeline.structs import BottleCandidates
from vinishko.pred.schemas import BottleOut
from vinishko.pred.schemas import CandidateOut
from vinishko.pred.schemas import HealthResponse
from vinishko.pred.schemas import ImageOut
from vinishko.pred.schemas import MatchOut
from vinishko.pred.schemas import RecognizeResponse
from vinishko.pred.schemas import RejectionOut
from vinishko.pred.schemas import ServiceErrorOut
from vinishko.whatis.service import RecognitionService


IMAGE_ROUTE = "/catalog/images/{name}"
AUTO_WHATIS = False
MAX_RECOGNITION_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 80_000_000

router = APIRouter(tags=["recognition"])


def polygons(polys: list | None) -> list[list[list[float]]] | None:
    """Полигоны маски в JSON-виде."""
    if polys is None:
        return None
    return [[[float(x), float(y)] for x, y in poly] for poly in polys]


def candidate_out(candidate: Candidate, catalog: Catalog) -> CandidateOut:
    """Позиция в ответе: скор поиска, картинка из коллекции и строка каталога; что все позиции коллекции есть в каталоге, Pipeline проверил при старте."""
    return CandidateOut(
        slug=candidate.slug,
        score=candidate.score,
        group=candidate.group,
        image_url=IMAGE_ROUTE.format(name=candidate.payload[FIELD_IMAGE]),
        catalog=catalog.get(candidate.slug),
    )


def bottle_out(bottle: BottleOutcome, catalog: Catalog) -> BottleOut:
    """Ответ по одной бутылке из итога пайплайна."""
    match = None
    if bottle.match is not None:
        match = MatchOut(
            **candidate_out(bottle.match.candidate, catalog).model_dump(),
            source=bottle.match.source,
            checklist=bottle.match.checklist,
        )
    rejection = None
    if bottle.rejection is not None:
        r = bottle.rejection
        rejection = RejectionOut(
            stage=r.stage,
            reason=r.reason.value,
            label=r.label,
            description=r.description,
            detail=r.detail,
            message=r.message,
        )
    if bottle.crop is None:
        raise ValueError("в ответ идут только бутылки, прошедшие нормализацию")
    return BottleOut(
        uuid=bottle.uuid,
        index=bottle.crop.index,
        score=bottle.score,
        polygons=polygons(bottle.bottle) or [],
        label_polygons=polygons(bottle.label),
        bbox=bottle.bbox,
        status=bottle.status,
        match=match,
        candidates=[candidate_out(c, catalog) for c in bottle.candidates],
        rejection=rejection,
    )


def recognize_response(
    image: Image.Image, result: PipelineResult, catalog: Catalog
) -> RecognizeResponse:
    """Ответ на фото: бутылки, прошедшие нормализацию, и число отброшенных ею."""
    passed = [b for b in result.bottles if b.crop is not None]
    return RecognizeResponse(
        image=ImageOut(width=image.width, height=image.height),
        bottles=[bottle_out(b, catalog) for b in passed],
        ignored=len(result.bottles) - len(passed),
        timings_s={k: round(v, 3) for k, v in result.timings.items()},
    )


async def attach_unknown_wine(
    response: RecognizeResponse, result: PipelineResult, whatis: RecognitionService
) -> None:
    """При AUTO_WHATIS — категория и винодельня от whatis по box_crop каждой отвергнутой бутылки, по очереди; ошибка whatis ложится в unknown_wine_error, исход распознавания не меняется."""
    rejected = [b for b in response.bottles if b.status == "rejected"]
    if (
        not (
            AUTO_WHATIS
            or os.getenv("AUTO_WHATIS", "false").lower() in {"true", "1", "yes"}
        )
        or not rejected
    ):
        return
    crops = {b.uuid: b.crop for b in result.bottles if b.crop is not None}
    started = time.perf_counter()
    for bottle in rejected:
        data = await asyncio.to_thread(
            encode_image, crops[bottle.uuid].box_crop, "jpg", 90
        )
        try:
            bottle.unknown_wine = await whatis.recognize(data)
        except HTTPException as error:
            bottle.unknown_wine_error = ServiceErrorOut(
                status_code=error.status_code, detail=str(error.detail)
            )
    response.timings_s["whatis"] = round(time.perf_counter() - started, 3)


def current(request: Request) -> Pipeline:
    return cast(Pipeline, request.app.state.pipeline)


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Что поднято: коллекция, модель поиска, второй уровень, каталог."""
    pipe = current(request)
    searcher = pipe.searcher if isinstance(pipe.searcher, VisSearcher) else None
    resolver = (
        pipe.resolver if isinstance(pipe.resolver, NearDuplicateResolver) else None
    )
    return HealthResponse(
        status="ok",
        collection=searcher.info.name if searcher else None,
        model=f"{searcher.files.repo}@{searcher.files.revision[:8]}"
        if searcher
        else None,
        encoder_input=list(searcher.cfg.encoder_input) if searcher else [],
        search_mode=searcher.cfg.search.mode if searcher else None,
        resolver=(resolver.settings.openrouter.model or "OPENROUTER_MODEL")
        if resolver
        else None,
        catalog=pipe.catalog.description,
        catalog_rows=len(pipe.catalog),
    )


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize(
    request: Request,
    image: Annotated[UploadFile, File(description="Фото: jpeg, png, webp, heic")],
) -> RecognizeResponse:
    """Фото → бутылки с исходом."""
    data = await image.read(MAX_RECOGNITION_BYTES + 1)
    if len(data) > MAX_RECOGNITION_BYTES:
        raise HTTPException(status_code=413, detail="Фото превышает 20 МиБ")

    def decode() -> Image.Image:
        with Image.open(BytesIO(data)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413, detail="Фото превышает 80 мегапикселей"
                )
        return open_image(BytesIO(data))

    try:
        picture = await asyncio.to_thread(decode)
    except Image.DecompressionBombError as error:
        raise HTTPException(
            status_code=413, detail="Слишком большое разрешение фото"
        ) from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(
            status_code=400, detail=f"файл не читается как изображение: {error}"
        ) from error
    pipe = current(request)
    lock: threading.Lock = request.app.state.lock

    def run() -> PipelineResult:
        with lock:
            return pipe(picture)

    result = await asyncio.to_thread(run)
    response = recognize_response(picture, result, pipe.catalog)
    if request.url.path == "/api/recognize":
        found = {
            b.crop.uuid: b.candidates
            for b in result.search
            if isinstance(b, BottleCandidates)
        }
        for bottle in response.bottles:
            bottle.candidates = [
                candidate_out(c, pipe.catalog) for c in found.get(bottle.uuid, [])
            ]
    await attach_unknown_wine(response, result, request.app.state.whatis)
    return response


@router.get(IMAGE_ROUTE)
async def catalog_image(request: Request, name: str) -> Response:
    """Картинка позиции из хранилища коллекции, JPEG."""
    pipe = current(request)
    if not isinstance(pipe.searcher, VisSearcher):
        raise HTTPException(
            status_code=404, detail="поиск выключен, картинок коллекции нет"
        )
    store = pipe.searcher.store
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise HTTPException(status_code=404, detail="Изображение не найдено")
    if not await asyncio.to_thread(store.exists, name):
        raise HTTPException(status_code=404, detail=f"в хранилище нет {name}")
    picture = await asyncio.to_thread(store.get, name)
    return Response(
        content=await asyncio.to_thread(encode_image, picture, "jpg", 90),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )
