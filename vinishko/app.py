"""Главное приложение vinishko: все ручки в одном процессе.

Распознавание — vinishko.pred (POST /recognize, GET /catalog/images/{name}, GET /health), сомелье — vinishko.sommelier
(/sommelier/sessions/...), признаки вина вне каталога — vinishko.whatis (POST /whatis). При старте читается корневой .env, поднимаются
пайплайн и сервисы сомелье и whatis. Сомелье и whatis ходят в OpenRouter и замка пайплайна не ждут.

Запуск из корня: python -m vinishko.app [--host 127.0.0.1] [--port 8000]; конфиги шагов — как у пайплайна, флаги ниже их перекрывают.
uvicorn vinishko.app:app — то же с конфигами по умолчанию.
"""

import asyncio
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import rich_click as click
import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from kostyl.utils import setup_logger

from vinishko.pred.pipeline.catalog import load_catalog
from vinishko.pred.pipeline.configs import DEFAULT_CONFIG as PIPELINE_CONFIG
from vinishko.pred.pipeline.configs import load_config as load_pipeline_config
from vinishko.pred.pipeline.pipeline import Pipeline
from vinishko.pred.pipeline.steps.near_duplicates.predictor import load_openrouter_env
from vinishko.pred.pipeline.steps.normalization.normalize import HERE as NORM_DIR
from vinishko.pred.pipeline.steps.normalization.normalize import Normalizer
from vinishko.pred.pipeline.steps.normalization.normalize import (
    load_config as load_norm_config,
)
from vinishko.pred.pipeline.steps.vis_searcher import VisSearcher
from vinishko.pred.pipeline.steps.vis_searcher.configs import (
    DEFAULT_CONFIG as SEARCH_CONFIG,
)
from vinishko.pred.pipeline.steps.vis_searcher.configs import (
    load_config as load_search_config,
)
from vinishko.pred.router import router as pred_router
from vinishko.pred.frontend import catalog_images, router as frontend_router
from vinishko.pred.ui import WineCatalog
from vinishko.sommelier.router import router as sommelier_router
from vinishko.sommelier.service import SommelierService
from vinishko.sommelier.storage import JsonSessionStore
from vinishko.whatis.router import router as whatis_router
from vinishko.whatis.service import RecognitionService


logger = setup_logger(fmt="detailed")
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend" / "dist"


def default_pipeline() -> Pipeline:
    max_bottles = int(os.getenv("MAX_BOTTLES", "10"))
    if not 1 <= max_bottles <= 100:
        raise ValueError("MAX_BOTTLES должен быть от 1 до 100")
    config = load_norm_config(
        NORM_DIR / "normalize.toml",
        [f"selection.max_bottles={max_bottles}"],
    )
    return Pipeline(normalizer=Normalizer(config, NORM_DIR))


def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    """Приложение; без pipeline он поднимается при старте с конфигами по умолчанию."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        load_openrouter_env()
        sessions = Path(os.environ.get("SOMMELIER_DATA_DIR", "data/sessions"))
        app.state.sommelier = SommelierService(JsonSessionStore(sessions.resolve()))
        app.state.whatis = RecognitionService()
        app.state.pipeline = (
            pipeline
            if pipeline is not None
            else await asyncio.to_thread(default_pipeline)
        )
        app.state.lock = threading.Lock()
        app.state.ui_catalog = WineCatalog(
            app.state.pipeline.catalog.rows,
            await asyncio.to_thread(catalog_images, app.state.pipeline),
        )
        logger.info("приложение готово")
        yield

    app = FastAPI(
        title="vinishko",
        description="Распознавание вина по фото бутылки, сомелье и признаки вина вне каталога",
        lifespan=lifespan,
    )
    app.include_router(pred_router)
    app.include_router(sommelier_router)
    app.include_router(whatis_router)
    app.include_router(frontend_router)
    app.include_router(sommelier_router, prefix="/api")
    app.include_router(whatis_router, prefix="/api")
    if (FRONTEND_DIR / "assets").is_dir():
        app.mount(
            "/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets"
        )

    @app.get("/", include_in_schema=False)
    @app.get("/scan/{scan_id}", include_in_schema=False)
    @app.get("/wine/{slug}", include_in_schema=False)
    async def frontend() -> FileResponse:
        index = FRONTEND_DIR / "index.html"
        if not index.is_file():
            raise HTTPException(
                status_code=503,
                detail="Откройте frontend на порту 8080 или соберите frontend/dist",
            )
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return app


app = create_app()


@click.command()
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="0.0.0.0 — слушать все интерфейсы",
)
@click.option("--port", type=int, default=8000, show_default=True)
@click.option(
    "--max-bottles",
    type=click.IntRange(1, 100),
    default=10,
    envvar="MAX_BOTTLES",
    show_default=True,
)
@click.option(
    "--pipeline-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=PIPELINE_CONFIG,
    show_default=True,
    help="Конфиг оркестратора: каталог",
)
@click.option(
    "--search-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=SEARCH_CONFIG,
    show_default=True,
    help="Конфиг визуального поиска",
)
@click.option(
    "-c",
    "--norm-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=NORM_DIR / "normalize.toml",
    show_default=True,
    help="Конфиг нормализации",
)
@click.option(
    "--no-resolve",
    is_flag=True,
    help="Без второго уровня: в ответе кандидаты поиска вместо выбранной позиции",
)
def main(
    host: str,
    port: int,
    max_bottles: int,
    pipeline_config: Path,
    search_config: Path,
    norm_config: Path,
    no_resolve: bool,
) -> None:
    """Поднять пайплайн и HTTP-сервер."""
    load_openrouter_env()
    cfg = load_pipeline_config(pipeline_config)
    search_cfg = load_search_config(search_config)
    search_cfg.debug_path = None
    pipeline = Pipeline(
        Normalizer(
            load_norm_config(norm_config, [f"selection.max_bottles={max_bottles}"]),
            norm_config.resolve().parent,
        ),
        VisSearcher(search_cfg),
        catalog=load_catalog(cfg.catalog, cfg.slug_column),
        resolve=not no_resolve,
    )
    uvicorn.run(create_app(pipeline), host=host, port=port)


if __name__ == "__main__":
    load_openrouter_env()
    main()
