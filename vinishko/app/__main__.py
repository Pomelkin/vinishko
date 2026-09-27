"""Запуск API: python -m vinishko.app [--host 127.0.0.1] [--port 8000]. Конфиги шагов — как у пайплайна, флаги ниже их перекрывают."""

from pathlib import Path

import rich_click as click
import uvicorn

from vinishko.app.main import create_app
from vinishko.pipeline.catalog import load_catalog
from vinishko.pipeline.configs import DEFAULT_CONFIG as PIPELINE_CONFIG
from vinishko.pipeline.configs import load_config as load_pipeline_config
from vinishko.pipeline.pipeline import Pipeline
from vinishko.pipeline.steps.normalization.normalize import HERE as NORM_DIR
from vinishko.pipeline.steps.normalization.normalize import Normalizer
from vinishko.pipeline.steps.normalization.normalize import (
    load_config as load_norm_config,
)
from vinishko.pipeline.steps.vis_searcher import VisSearcher
from vinishko.pipeline.steps.vis_searcher.configs import DEFAULT_CONFIG as SEARCH_CONFIG
from vinishko.pipeline.steps.vis_searcher.configs import (
    load_config as load_search_config,
)


@click.command()
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="0.0.0.0 — слушать все интерфейсы",
)
@click.option("--port", type=int, default=8000, show_default=True)
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
    pipeline_config: Path,
    search_config: Path,
    norm_config: Path,
    no_resolve: bool,
) -> None:
    """Поднять пайплайн и HTTP-сервер."""
    cfg = load_pipeline_config(pipeline_config)
    search_cfg = load_search_config(search_config)
    search_cfg.debug_path = None
    pipeline = Pipeline(
        Normalizer(load_norm_config(norm_config, []), norm_config.resolve().parent),
        VisSearcher(search_cfg),
        catalog=load_catalog(cfg.catalog, cfg.slug_column),
        resolve=not no_resolve,
    )
    uvicorn.run(create_app(pipeline), host=host, port=port)


if __name__ == "__main__":
    main()
