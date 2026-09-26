"""Сборка коллекции каталога: CSV → нормализация каждого фото → кроп в хранилище → вектор в qdrant.

Модель, qdrant с именем коллекции и хранилище кропов берутся из того же config.yaml, с которым потом ищет VisSearcher: собрать одним, а искать
другим нельзя по построению. Запуск из корня: python -m vinishko.pipeline.steps.vis_searcher.build_catalog --csv ... --images ...
"""

import csv
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rich_click as click
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from kostyl.utils import setup_logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from vinishko.pipeline.steps.normalization.normalize import HERE as NORM_DIR
from vinishko.pipeline.steps.normalization.normalize import Normalizer
from vinishko.pipeline.steps.normalization.normalize import (
    load_config as load_norm_config,
)
from vinishko.pipeline.steps.normalization.seg import open_image
from vinishko.pipeline.steps.vis_searcher.catalog import (
    FIELD_GROUP,
    FIELD_GROUP_SLUGS,
    FIELD_IMAGE,
    FIELD_INPUT_SIZE,
    FIELD_MODEL,
    FIELD_REVISION,
    FIELD_SLUG,
    connect,
    create_collection,
    point_id,
    upsert,
)
from vinishko.pipeline.steps.vis_searcher.configs import DEFAULT_CONFIG, load_config
from vinishko.pipeline.steps.vis_searcher.device import Device, resolve_device
from vinishko.pipeline.steps.vis_searcher.model import Encoder, ModelFiles, fetch_model
from vinishko.pipeline.steps.vis_searcher.storage import (
    CONTENT_TYPES,
    ImageStore,
    make_store,
)
from vinishko.pipeline.structs import BottleCrop

CSV_FIELDS = {
    "name": "Название вина",
    "category": "Категория",
    "color": "Цвет",
    "region": "Регион",
    "grape": "Сорт винограда",
    "winery": "Винодельня",
    "vintage": "Винтаж",
    "abv": "Крепость, %",
    "sugar": "Сахар",
    "sparkling": "Игристое",
}
"""Колонки каталога, которые уходят в метаданные точки под латинскими именами: второму уровню нужны поля, а не весь CSV."""
console = Console()
logger = setup_logger(fmt="detailed")


@dataclass(slots=True)
class Row:
    """Строка каталога, у которой есть фото."""

    slug: str
    photo: str
    group: str
    fields: dict[str, str]


@dataclass(slots=True)
class Failure:
    """Фото, не попавшее в коллекцию."""

    slug: str
    photo: str
    reason: str


def read_rows(
    path: Path,
    slug_column: str,
    photo_column: str,
    group_column: str,
    limit: int | None,
) -> tuple[list[Row], int, bool]:
    """Строки с фото, сколько строк без фото пропущено и была ли колонка групп; без неё каждая позиция — своя группа."""
    with path.open(encoding="utf-8-sig", newline="") as f:  # utf-8-sig: у выгрузок бывает BOM, иначе он прилипает к имени первой колонки
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        for column in (slug_column, photo_column):
            if column not in columns:
                raise click.UsageError(
                    f"в {path.name} нет колонки {column!r}; есть {columns[:6]}…"
                )
        has_group = group_column in columns
        rows, skipped = [], 0
        for rec in reader:
            slug, photo = rec[slug_column].strip(), (rec[photo_column] or "").strip()
            if not slug:
                raise ValueError(f"{path.name}: строка с пустым {slug_column}")
            if not photo:
                skipped += 1
                continue
            group = (rec.get(group_column) or "").strip() if has_group else ""
            fields = {
                key: rec[column].strip()
                for key, column in CSV_FIELDS.items()
                if rec.get(column) and rec[column].strip()
            }
            rows.append(Row(slug, photo, group or slug, fields))
    duplicates = sorted(
        {r.slug for r in rows if sum(x.slug == r.slug for x in rows) > 1}
    )
    if duplicates:
        raise ValueError(f"{path.name}: slug повторяются, например {duplicates[:5]}")
    return rows[:limit], skipped, has_group


def check_photos(rows: list[Row], images: Path) -> None:
    """Все фото из CSV лежат в images; иначе ошибка до старта, а не через час работы."""
    missing = [row.photo for row in rows if not (images / row.photo).is_file()]
    if missing:
        raise click.UsageError(f"в {images} нет {len(missing)} фото из CSV, например {missing[:3]}")


def ensure_absent(client: QdrantClient, name: str) -> None:
    """Существующая коллекция удаляется только с подтверждением в терминале; без терминала это ошибка."""
    if not client.collection_exists(name):
        return
    if not sys.stdin.isatty():
        raise RuntimeError(
            f"коллекция {name} уже есть; без интерактивного терминала удалить её надо отдельно"
        )
    if not click.confirm(
        f"коллекция {name} уже есть, удалить и собрать заново?", default=False
    ):
        raise RuntimeError(f"коллекция {name} оставлена как есть")
    client.delete_collection(name)


def single_crop(norm: Normalizer, path: Path) -> BottleCrop:
    """Единственная годная бутылка на фото каталога; иначе ошибка с причинами отказов."""
    items = norm(open_image(path))
    crops = [item for item in items if isinstance(item, BottleCrop)]
    if len(crops) != 1:
        reasons = ", ".join(
            item.reason for item in items if not isinstance(item, BottleCrop)
        )
        raise ValueError(
            f"годных бутылок {len(crops)}, нужна ровно одна"
            + (f"; отказы: {reasons}" if reasons else "")
        )
    return crops[0]


def make_payload(
    row: Row, crop: BottleCrop, image: str, files: ModelFiles, device: Device
) -> dict:
    """Метаданные точки; список позиций группы дописывается, когда известно, кто из группы попал в коллекцию."""
    return {
        FIELD_SLUG: row.slug,
        FIELD_GROUP: row.group,
        FIELD_GROUP_SLUGS: [],
        FIELD_IMAGE: image,
        "source_image": row.photo,
        **row.fields,
        "bottle_score": crop.score,
        "bottle_angle": crop.angle,
        FIELD_MODEL: files.repo,
        FIELD_REVISION: files.revision,
        FIELD_INPUT_SIZE: list(files.input_size),
        "precision": device.precision,
        "built_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
    }


def batched(rows: Sequence[Row], size: int) -> Iterator[Sequence[Row]]:
    """Строки пачками."""
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def progress() -> Progress:
    """Полоса прогресса."""
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


@dataclass(slots=True)
class Build:
    """Ход сборки: векторы и метаданные по slug, неудачи."""

    vectors: dict[str, np.ndarray]
    payloads: dict[str, dict]
    failures: list[Failure]

    def points(self) -> list[PointStruct]:
        """Точки с заполненными списками групп: в группе только те позиции, что попали в коллекцию."""
        members: dict[str, list[str]] = {}
        for slug, payload in self.payloads.items():
            members.setdefault(payload[FIELD_GROUP], []).append(slug)
        for payload in self.payloads.values():
            payload[FIELD_GROUP_SLUGS] = members[payload[FIELD_GROUP]]
        return [
            PointStruct(
                id=point_id(slug),
                vector=self.vectors[slug].tolist(),
                payload=self.payloads[slug],
            )
            for slug in self.payloads
        ]


def build(
    rows: list[Row],
    images: Path,
    norm: Normalizer,
    encoder: Encoder,
    store: ImageStore,
    fmt: str,
    quality: int,
    skip_failures: bool,
) -> Build:
    """Фото → кроп → хранилище и вектор, пачками по батчу энкодера."""
    state = Build({}, {}, [])
    with progress() as bar:
        task = bar.add_task("каталог", total=len(rows))
        for batch in batched(rows, encoder.max_batch):
            crops: list[tuple[Row, BottleCrop]] = []
            for row in batch:
                if skip_failures:
                    try:
                        crops.append((row, single_crop(norm, images / row.photo)))
                    except Exception as error:  # одна битая картинка не должна ронять сборку на час, отчёт о ней будет в конце
                        state.failures.append(Failure(row.slug, row.photo, repr(error)))
                else:
                    crops.append((row, single_crop(norm, images / row.photo)))
            vectors = encoder([crop.crop for _, crop in crops])
            for (row, crop), vector in zip(crops, vectors, strict=True):
                name = f"{row.slug}.{fmt}"
                store.put(name, crop.crop, fmt, quality)
                state.vectors[row.slug] = vector
                state.payloads[row.slug] = make_payload(
                    row, crop, name, encoder.files, encoder.device
                )
            bar.update(task, advance=len(batch))
    return state


def report(
    state: Build,
    rows: list[Row],
    skipped: int,
    has_group: bool,
    collection: str,
    path: Path | None,
) -> None:
    """Итог в терминал и, если задан путь, в JSON."""
    table = Table(title=f"коллекция {collection}")
    table.add_column("что")
    table.add_column("сколько", justify="right")
    groups = {p[FIELD_GROUP] for p in state.payloads.values()}
    for name, value in (
        ("строк с фото", len(rows)),
        ("строк без фото, пропущено", skipped),
        ("в коллекции", len(state.payloads)),
        (
            "групп" if has_group else "групп (колонки нет, позиция = группа)",
            len(groups),
        ),
        ("не попало", len(state.failures)),
    ):
        table.add_row(name, str(value))
    console.print(table)
    for failure in state.failures[:20]:
        logger.warning(f"не попало: {failure.slug}: {failure.photo}: {failure.reason}")
    if len(state.failures) > 20:
        logger.warning(f"… и ещё {len(state.failures) - 20} неудач, полный список в --report")
    if path is not None:
        path.write_text(
            json.dumps(
                {
                    "collection": collection,
                    "indexed": len(state.payloads),
                    "skipped_no_photo": skipped,
                    "failures": [asdict(f) for f in state.failures],
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        logger.info(f"отчёт: {path}")


@click.command()
@click.option("--csv", "csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Каталог: CSV с колонками slug, имени фото и группы")
@click.option("--images", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True, help="Директория с фото из CSV")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=DEFAULT_CONFIG, show_default=True, help="Конфиг визуального поиска: модель и ревизия, qdrant и имя коллекции, хранилище кропов, batch_size, cache_dir")
@click.option("--slug-column", default="Slug", show_default=True)
@click.option("--photo-column", default="Название фото", show_default=True)
@click.option("--group-column", default="group", show_default=True, help="Колонка группы одинакового дизайна; если её нет, каждая позиция — своя группа")
@click.option("--format", "fmt", type=click.Choice(sorted(CONTENT_TYPES)), default="jpg", show_default=True, help="Формат кропов в хранилище")
@click.option("--quality", type=click.IntRange(1, 100), default=95, show_default=True)
@click.option("--on-failure", type=click.Choice(["stop", "skip"]), default="stop", show_default=True, help="Фото без ровно одной годной бутылки: остановить сборку либо пропустить и перечислить в конце")
@click.option("--limit", type=click.IntRange(min=1), default=None, help="Первые N строк с фото: для пробы")
@click.option("--report", "report_path", type=click.Path(dir_okay=False, path_type=Path), default=None, help="JSON с итогом и списком неудач")
@click.option("-c", "--norm-config", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=NORM_DIR / "normalize.toml", show_default=True, help="Конфиг нормализации")
@click.option("--set", "overrides", multiple=True, metavar="секция.ключ=значение", help="Переопределить параметр конфига нормализации")
def main(
    csv_path: Path,
    images: Path,
    config_path: Path,
    slug_column: str,
    photo_column: str,
    group_column: str,
    fmt: str,
    quality: int,
    on_failure: str,
    limit: int | None,
    report_path: Path | None,
    norm_config: Path,
    overrides: tuple[str, ...],
) -> None:
    """Собрать коллекцию каталога для визуального поиска по config.yaml.

    Каждое фото проходит нормализацию, на нём должна найтись ровно одна годная бутылка; её кроп уходит в хранилище картинок из конфига,
    вектор — в коллекцию qdrant из конфига вместе с метаданными: slug, группа и все slug группы, попавшие в коллекцию, имя кропа,
    поля каталога, модель и её ревизия. Устройства — device в конфигах, переменные VIS_SEARCHER_DEV и NORMALIZER_DEV их перекрывают.
    Дешёвые проверки идут до загрузки моделей: CSV и наличие всех фото, qdrant и отсутствие коллекции, доступность хранилища.
    """
    cfg = load_config(config_path)
    rows, skipped, has_group = read_rows(csv_path, slug_column, photo_column, group_column, limit)
    check_photos(rows, images)
    device = resolve_device(cfg.device)
    files = fetch_model(cfg.model, device.precision, cfg.revision)
    client = connect(cfg.qdrant)
    collection = cfg.qdrant.collection
    ensure_absent(client, collection)
    store = make_store(cfg.images, cfg.cache_dir)
    store.ensure_available()
    norm = Normalizer(load_norm_config(norm_config, list(overrides)), norm_config.resolve().parent)
    logger.info(
        f"конфиг {config_path}: модель {cfg.model}, коллекция {collection}; каталог: {len(rows)} строк с фото, {skipped} без фото; "
        f"группы: {'колонка ' + group_column if has_group else 'нет, позиция = группа'}; кропы: {store.description}; нормализация на {norm.device}"
    )
    encoder = Encoder(files, device, cfg.batch_size, cfg.cache_dir)
    logger.info(f"энкодер: {encoder.description}")
    create_collection(client, collection, encoder.embed_dim)
    state = build(rows, images, norm, encoder, store, fmt, quality, skip_failures=on_failure == "skip")
    upsert(client, collection, state.points())
    report(state, rows, skipped, has_group, collection, report_path)


if __name__ == "__main__":
    main()
