"""Снапшот готовой коллекции qdrant в хранилище снапшотов из config.yaml: python -m vinishko.pred.pipeline.steps.vis_searcher.dump_collection [--collection …] [--config …].

Поиск, не найдя коллекцию в qdrant, восстанавливает её оттуда при старте.
"""

import sys
from pathlib import Path

import rich_click as click
from kostyl.utils import setup_logger

from vinishko.pred.pipeline.steps.vis_searcher.catalog import connect
from vinishko.pred.pipeline.steps.vis_searcher.catalog import inspect
from vinishko.pred.pipeline.steps.vis_searcher.configs import DEFAULT_CONFIG
from vinishko.pred.pipeline.steps.vis_searcher.configs import load_config
from vinishko.pred.pipeline.steps.vis_searcher.snapshots import SnapshotStore
from vinishko.pred.pipeline.steps.vis_searcher.snapshots import dump_collection
from vinishko.pred.pipeline.steps.vis_searcher.snapshots import make_snapshot_store
from vinishko.pred.pipeline.steps.vis_searcher.snapshots import passport_name
from vinishko.pred.pipeline.steps.vis_searcher.storage import MANIFEST
from vinishko.pred.pipeline.steps.vis_searcher.storage import make_store


logger = setup_logger(fmt="detailed")


def ensure_replaceable(store: SnapshotStore, collection: str) -> None:
    """Снапшот коллекции, который уже лежит в хранилище, заменяется только с подтверждением в терминале; без терминала это ошибка."""
    if not store.exists(passport_name(collection)):
        return
    old = store.get_json(passport_name(collection))
    what = f"в {store.description} уже есть снапшот {collection} от {old['dumped_at']}, {old['points']} точек"
    if not sys.stdin.isatty():
        raise RuntimeError(f"{what}; без интерактивного терминала заменить его нельзя")
    if not click.confirm(f"{what}. Заменить?", default=False):
        raise RuntimeError(f"снапшот {collection} оставлен как есть")


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_CONFIG,
    show_default=True,
    help="Конфиг поиска: qdrant, хранилище снапшотов snapshots, хранилище картинок для сверки",
)
@click.option(
    "--collection",
    default=None,
    help="Коллекция; по умолчанию qdrant.collection из конфига",
)
def main(config_path: Path, collection: str | None) -> None:
    """Снять снапшот коллекции с сервера qdrant и положить его в хранилище снапшотов вместе с паспортом.

    Снапшот несёт HNSW-граф, восстановление на пустом qdrant занимает секунды и не требует ни GPU, ни фото каталога. Отпечаток нормализации
    коллекции сверяется с manifest.json хранилища картинок: если сборки разные, снапшот снимется, но поиск с этим хранилищем его не примет.
    """
    cfg = load_config(config_path)
    if cfg.snapshots is None:
        raise click.UsageError(f"в {config_path} нет snapshots: некуда класть снапшот")
    name = collection or cfg.qdrant.collection
    store = make_snapshot_store(cfg.snapshots, cfg.cache_dir)
    store.ensure_available()
    ensure_replaceable(store, name)
    client = connect(cfg.qdrant)
    info = inspect(client, name)
    images = make_store(cfg.images, cfg.cache_dir)
    if images.exists(MANIFEST):
        manifest = images.get_json(MANIFEST)
        if manifest["fingerprint"] != info.normalization:
            logger.warning(
                f"коллекция {name} собрана с нормализацией {info.normalization}, а картинки в {images.description} — сборка {manifest['collection']} "
                f"с {manifest['fingerprint']}: поиск с этим хранилищем восстановленную коллекцию не примет"
            )
    else:
        logger.warning(
            f"в {images.description} нет {MANIFEST}: поиск с этим хранилищем картинок не поднимется"
        )
    logger.info(
        f"коллекция {name}: {info.count} точек ×{info.size}, {info.model}@{info.revision[:8]}, входы {'+'.join(info.encoder_input)}; "
        f"снапшот → {store.description}"
    )
    dump_collection(
        client,
        cfg.qdrant,
        info,
        store,
        cfg.cache_dir / "snapshots" / "dump",
        cfg.snapshots.timeout,
    )


if __name__ == "__main__":
    main()
