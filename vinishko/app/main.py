"""FastAPI поверх пайплайна.

POST /recognize принимает фото и отдаёт список бутылок, дошедших до поиска: маска на исходном фото, скор отбора и исход — выбранная
позиция каталога (slug, скор, строка каталога, картинка из коллекции) либо отказ поиска или второго уровня с причиной. Объекты, отброшенные
нормализацией (не целевая бутылка, нет читаемой этикетки), в список не входят, только их число. GET /catalog/images/<имя> отдаёт картинку позиции,
GET /health — состояние. Пайплайн поднимается при старте приложения; запросы к нему идут по одному, под замком: SAM3 и engine
TensorRT не рассчитаны на параллельные вызовы из одного процесса. Запуск: python -m vinishko.app.
"""

import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated, cast

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from kostyl.utils import setup_logger
from PIL import Image, UnidentifiedImageError

from vinishko.app.schemas import (
    BottleOut,
    CandidateOut,
    HealthResponse,
    ImageOut,
    MatchOut,
    RecognizeResponse,
    RejectionOut,
)
from vinishko.pipeline.catalog import Catalog
from vinishko.pipeline.pipeline import Pipeline, PipelineResult
from vinishko.pipeline.steps.near_duplicates import NearDuplicateResolver
from vinishko.pipeline.steps.normalization.seg import open_image
from vinishko.pipeline.steps.vis_searcher.catalog import FIELD_IMAGE
from vinishko.pipeline.steps.vis_searcher.search import VisSearcher
from vinishko.pipeline.steps.vis_searcher.storage import encode_image
from vinishko.pipeline.structs import BottleOutcome

logger = setup_logger(fmt="detailed")
IMAGE_ROUTE = "/catalog/images/{name}"


def polygons(polys: list | None) -> list[list[list[float]]] | None:
    """Полигоны маски в JSON-виде."""
    if polys is None:
        return None
    return [[[float(x), float(y)] for x, y in poly] for poly in polys]


def bottle_out(bottle: BottleOutcome, catalog: Catalog) -> BottleOut:
    """Ответ по одной бутылке из итога пайплайна."""
    match = None
    if bottle.match is not None:
        position = bottle.match.candidate
        if position.slug not in catalog:
            logger.warning(f"позиции {position.slug} нет в каталоге ({catalog.description}): коллекция и каталог из разных выгрузок")
        match = MatchOut(
            slug=position.slug,
            score=position.score,
            source=bottle.match.source,
            group=position.group,
            image_url=IMAGE_ROUTE.format(name=position.payload[FIELD_IMAGE]),
            catalog=catalog.rows.get(position.slug, {}),
            checklist=bottle.match.checklist,
        )
    rejection = None
    if bottle.rejection is not None:
        r = bottle.rejection
        rejection = RejectionOut(stage=r.stage, reason=r.reason.value, label=r.label, description=r.description, detail=r.detail, message=r.message)
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
        candidates=[CandidateOut(slug=c.slug, score=c.score, group=c.group) for c in bottle.candidates],
        rejection=rejection,
    )


def recognize_response(image: Image.Image, result: PipelineResult, catalog: Catalog) -> RecognizeResponse:
    """Ответ на фото: бутылки, прошедшие нормализацию, и число отброшенных ею."""
    passed = [b for b in result.bottles if b.crop is not None]
    return RecognizeResponse(
        image=ImageOut(width=image.width, height=image.height),
        bottles=[bottle_out(b, catalog) for b in passed],
        ignored=len(result.bottles) - len(passed),
        timings_s={k: round(v, 3) for k, v in result.timings.items()},
    )


def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    """Приложение; без pipeline он поднимается при старте с конфигами по умолчанию."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pipeline = pipeline if pipeline is not None else await asyncio.to_thread(Pipeline)
        app.state.lock = threading.Lock()
        logger.info("приложение готово")
        yield

    app = FastAPI(title="vinishko", description="Распознавание вина по фото бутылки", lifespan=lifespan)

    def current(request: Request) -> Pipeline:
        return cast(Pipeline, request.app.state.pipeline)

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        """Что поднято: коллекция, модель поиска, второй уровень, каталог."""
        pipe = current(request)
        searcher = pipe.searcher if isinstance(pipe.searcher, VisSearcher) else None
        resolver = pipe.resolver if isinstance(pipe.resolver, NearDuplicateResolver) else None
        return HealthResponse(
            status="ok",
            collection=searcher.info.name if searcher else None,
            model=f"{searcher.files.repo}@{searcher.files.revision[:8]}" if searcher else None,
            encoder_input=list(searcher.cfg.encoder_input) if searcher else [],
            search_mode=searcher.cfg.search.mode if searcher else None,
            resolver=(resolver.settings.openrouter.model or "OPENROUTER_MODEL") if resolver else None,
            catalog=pipe.catalog.description,
            catalog_rows=len(pipe.catalog),
        )

    @app.post("/recognize", response_model=RecognizeResponse)
    async def recognize(request: Request, image: Annotated[UploadFile, File(description="Фото: jpeg, png, webp, heic")]) -> RecognizeResponse:
        """Фото → бутылки с исходом."""
        data = await image.read()
        try:
            picture = open_image(BytesIO(data))
        except (UnidentifiedImageError, OSError) as error:
            raise HTTPException(status_code=400, detail=f"файл не читается как изображение: {error}") from error
        pipe = current(request)
        lock: threading.Lock = request.app.state.lock

        def run() -> PipelineResult:
            with lock:
                return pipe(picture)

        result = await asyncio.to_thread(run)
        return recognize_response(picture, result, pipe.catalog)

    @app.get(IMAGE_ROUTE)
    async def catalog_image(request: Request, name: str) -> Response:
        """Картинка позиции из хранилища коллекции, JPEG."""
        pipe = current(request)
        if not isinstance(pipe.searcher, VisSearcher):
            raise HTTPException(status_code=404, detail="поиск выключен, картинок коллекции нет")
        store = pipe.searcher.store
        if not await asyncio.to_thread(store.exists, name):
            raise HTTPException(status_code=404, detail=f"в хранилище нет {name}")
        picture = await asyncio.to_thread(store.get, name)
        return Response(content=encode_image(picture, "jpg", 90), media_type="image/jpeg")

    return app


app = create_app()
