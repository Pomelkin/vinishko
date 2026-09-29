"""Сборка коллекции каталога: CSV → нормализация каждого фото → кроп в хранилище → вектор в qdrant, по именованному вектору на каждый вход encoder_input.

Модель, qdrant с именем коллекции и хранилище кропов берутся из того же config.yaml, с которым потом ищет VisSearcher: собрать одним, а искать
другим нельзя по построению. Часть конфига нормализации, определяющая кропы (Normalizer.crop_config), записывается в manifest.json рядом
с картинками, её отпечаток — в каждую точку: поиск при старте сверяет с ними нормализатор запроса.
Запуск из корня: python -m vinishko.pred.pipeline.steps.vis_searcher.build_catalog --csv ... --images ...
"""

import csv
import json
import sys
from collections.abc import Iterator
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path

import numpy as np
import rich_click as click
from kostyl.utils import setup_logger
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from rich.console import Console
from rich.progress import BarColumn
from rich.progress import MofNCompleteColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from rich.table import Table

from vinishko.pred.pipeline.steps.normalization.normalize import HERE as NORM_DIR
from vinishko.pred.pipeline.steps.normalization.normalize import Normalizer
from vinishko.pred.pipeline.steps.normalization.normalize import crop_config
from vinishko.pred.pipeline.steps.normalization.normalize import (
    load_config as load_norm_config,
)
from vinishko.pred.pipeline.steps.normalization.seg import open_image
from vinishko.pred.pipeline.steps.vis_searcher.catalog import FIELD_ENCODER_INPUT
from vinishko.pred.pipeline.steps.vis_searcher.catalog import FIELD_GROUP
from vinishko.pred.pipeline.steps.vis_searcher.catalog import FIELD_GROUP_SLUGS
from vinishko.pred.pipeline.steps.vis_searcher.catalog import FIELD_IMAGE
from vinishko.pred.pipeline.steps.vis_searcher.catalog import FIELD_INPUT_SIZE
from vinishko.pred.pipeline.steps.vis_searcher.catalog import FIELD_MODEL
from vinishko.pred.pipeline.steps.vis_searcher.catalog import FIELD_NORMALIZATION
from vinishko.pred.pipeline.steps.vis_searcher.catalog import FIELD_REVISION
from vinishko.pred.pipeline.steps.vis_searcher.catalog import FIELD_SLUG
from vinishko.pred.pipeline.steps.vis_searcher.catalog import connect
from vinishko.pred.pipeline.steps.vis_searcher.catalog import create_collection
from vinishko.pred.pipeline.steps.vis_searcher.catalog import normalization_fingerprint
from vinishko.pred.pipeline.steps.vis_searcher.catalog import point_id
from vinishko.pred.pipeline.steps.vis_searcher.catalog import upsert
from vinishko.pred.pipeline.steps.vis_searcher.configs import DEFAULT_CONFIG
from vinishko.pred.pipeline.steps.vis_searcher.configs import load_config
from vinishko.pred.pipeline.steps.vis_searcher.device import Device
from vinishko.pred.pipeline.steps.vis_searcher.device import resolve_device
from vinishko.pred.pipeline.steps.vis_searcher.model import Encoder
from vinishko.pred.pipeline.steps.vis_searcher.model import ModelFiles
from vinishko.pred.pipeline.steps.vis_searcher.model import fetch_model
from vinishko.pred.pipeline.steps.vis_searcher.storage import CONTENT_TYPES
from vinishko.pred.pipeline.steps.vis_searcher.storage import MANIFEST
from vinishko.pred.pipeline.steps.vis_searcher.storage import ImageStore
from vinishko.pred.pipeline.steps.vis_searcher.storage import make_store
from vinishko.pred.pipeline.structs import BottleCrop


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
    "aging_or_reserve": "Выдержка или резерв",
}
"""Колонки каталога, которые уходят в метаданные точки под латинскими именами: второму уровню нужны поля, а не весь CSV.
aging_or_reserve добавлен 2026-09-28 для карточки NDR; в коллекциях, собранных раньше, поля нет, карточка получает пустую строку."""
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
class Issue:
    """Фото, не попавшее в коллекцию либо попавшее в неё всем кадром без нормализации."""

    slug: str
    photo: str
    reason: str


def read_rows(
    path: Path,
    slug_column: str,
    photo_column: str,
    group_column: str,
    limit: int | None,
) -> tuple[list[Row], list[str], bool]:
    """Строки с фото, slug строк без фото (они в коллекцию не идут) и была ли колонка групп; без неё каждая позиция — своя группа."""
    with (
        path.open(encoding="utf-8-sig", newline="") as f
    ):  # utf-8-sig: у выгрузок бывает BOM, иначе он прилипает к имени первой колонки
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        for column in (slug_column, photo_column):
            if column not in columns:
                raise click.UsageError(
                    f"в {path.name} нет колонки {column!r}; есть {columns[:6]}…"
                )
        has_group = group_column in columns
        rows, without = [], []
        for rec in reader:
            slug, photo = rec[slug_column].strip(), (rec[photo_column] or "").strip()
            if not slug:
                raise ValueError(f"{path.name}: строка с пустым {slug_column}")
            if not photo:
                without.append(slug)
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
    return rows[:limit], without, has_group


def check_photos(rows: list[Row], images: Path) -> None:
    """Все фото из CSV лежат в images; иначе ошибка до старта, а не через час работы."""
    missing = [row.photo for row in rows if not (images / row.photo).is_file()]
    if missing:
        raise click.UsageError(
            f"в {images} нет {len(missing)} фото из CSV, например {missing[:3]}"
        )


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


def catalog_crop(
    norm: Normalizer, images: Path, row: Row
) -> tuple[BottleCrop, str | None]:
    """Единственная годная бутылка на фото каталога и None; если нормализация не дала ровно одной, весь кадр (Normalizer.frame_crop) и причина
    с именем фото, slug и отказами: позиция каталога не теряется, её векторы считаются с фото как есть."""
    image = open_image(images / row.photo)
    items = norm(image)
    crops = [item for item in items if isinstance(item, BottleCrop)]
    if len(crops) == 1:
        return crops[0], None
    reasons = ", ".join(
        item.rejection.reason for item in items if not isinstance(item, BottleCrop)
    )
    return norm.frame_crop(image), (
        f"{row.photo} ({row.slug}): годных бутылок {len(crops)}, нужна ровно одна"
        + (f"; отказы: {reasons}" if reasons else "")
    )


def make_payload(
    row: Row,
    crop: BottleCrop,
    image: str,
    files: ModelFiles,
    device: Device,
    encoder_input: Sequence[str],
    normalization: str,
    whole_frame: bool,
) -> dict:
    """Метаданные точки; список позиций группы дописывается, когда известно, кто из группы попал в коллекцию. normalization — отпечаток crop_config.
    whole_frame — нормализация не нашла годной бутылки, картинка и векторы со всего кадра."""
    return {
        FIELD_SLUG: row.slug,
        FIELD_GROUP: row.group,
        FIELD_GROUP_SLUGS: [],
        FIELD_IMAGE: image,
        "image_crop": "whole_frame" if whole_frame else "bottle_box",
        "source_image": row.photo,
        **row.fields,
        "bottle_score": crop.score,
        "bottle_angle": crop.angle,
        FIELD_MODEL: files.repo,
        FIELD_REVISION: files.revision,
        FIELD_INPUT_SIZE: list(files.input_size),
        FIELD_ENCODER_INPUT: list(encoder_input),
        FIELD_NORMALIZATION: normalization,
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
    """Ход сборки: векторы по slug и входу энкодера, метаданные по slug, неудачи и позиции, попавшие всем кадром."""

    vectors: dict[str, dict[str, np.ndarray]]
    payloads: dict[str, dict]
    failures: list[Issue]
    whole_frames: list[Issue]

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
                vector={
                    inp: vector.tolist() for inp, vector in self.vectors[slug].items()
                },
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
    encoder_input: Sequence[str],
    normalization: str,
) -> Build:
    """Фото → кропы → хранилище и вектор на каждый вход энкодера, пачками по батчу энкодера. normalization — отпечаток crop_config, уходит в каждую точку.

    Фото без ровно одной годной бутылки идёт в коллекцию всем кадром с предупреждением, см. catalog_crop. Без skip_failures первая же
    ошибка чтения или нормализации останавливает сборку; с ним фото пропускается, а причина попадает в предупреждение и в отчёт.
    """
    state = Build({}, {}, [], [])
    with progress() as bar:
        task = bar.add_task("каталог", total=len(rows))
        for batch in batched(rows, encoder.max_batch):
            crops: list[tuple[Row, BottleCrop, str | None]] = []
            for row in batch:
                if not skip_failures:
                    crops.append((row, *catalog_crop(norm, images, row)))
                    continue
                try:
                    crops.append((row, *catalog_crop(norm, images, row)))
                except Exception as error:  # битая картинка не должна ронять сборку на час, отчёт о ней будет в конце
                    state.failures.append(
                        Issue(
                            row.slug, row.photo, f"{row.photo} ({row.slug}): {error!r}"
                        )
                    )
            vectors = {
                inp: encoder([getattr(crop, inp) for _, crop, _ in crops])
                for inp in encoder_input
            }
            for i, (row, crop, whole_frame) in enumerate(crops):
                if whole_frame is not None:
                    logger.warning(f"{whole_frame}; в коллекцию идёт весь кадр")
                    state.whole_frames.append(Issue(row.slug, row.photo, whole_frame))
                name = f"{row.slug}.{fmt}"
                store.put(
                    name, crop.box_crop, fmt, quality
                )  # векторы — с кропов по encoder_input, а в каталог идёт вся бутылка для второго уровня
                state.vectors[row.slug] = {
                    inp: vectors[inp][i] for inp in encoder_input
                }
                state.payloads[row.slug] = make_payload(
                    row,
                    crop,
                    name,
                    encoder.files,
                    encoder.device,
                    encoder_input,
                    normalization,
                    whole_frame is not None,
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
        ("из них всем кадром", len(state.whole_frames)),
        (
            "групп" if has_group else "групп (колонки нет, позиция = группа)",
            len(groups),
        ),
        ("не попало", len(state.failures)),
    ):
        table.add_row(name, str(value))
    console.print(table)
    for failure in state.failures[:20]:
        logger.warning(f"не попало: {failure.reason}")
    if len(state.failures) > 20:
        logger.warning(
            f"… и ещё {len(state.failures) - 20} неудач, полный список в --report"
        )
    if state.whole_frames:
        slugs = [issue.slug for issue in state.whole_frames]
        logger.warning(
            f"всем кадром без нормализации {len(slugs)}: {', '.join(slugs[:10])}"
            + (", … полный список в --report" if len(slugs) > 10 else "")
        )
    if path is not None:
        path.write_text(
            json.dumps(
                {
                    "collection": collection,
                    "indexed": len(state.payloads),
                    "skipped_no_photo": skipped,
                    "failures": [asdict(f) for f in state.failures],
                    "whole_frame": [asdict(f) for f in state.whole_frames],
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        logger.info(f"отчёт: {path}")


@click.command()
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Каталог: CSV с колонками slug, имени фото и группы",
)
@click.option(
    "--images",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Директория с фото из CSV",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_CONFIG,
    show_default=True,
    help="Конфиг визуального поиска: модель и ревизия, qdrant и имя коллекции, хранилище кропов, batch_size, cache_dir",
)
@click.option("--slug-column", default="Slug", show_default=True)
@click.option("--photo-column", default="Название фото", show_default=True)
@click.option(
    "--group-column",
    default="group",
    show_default=True,
    help="Колонка группы одинакового дизайна; если её нет, каждая позиция — своя группа",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(sorted(CONTENT_TYPES)),
    default="jpg",
    show_default=True,
    help="Формат кропов в хранилище",
)
@click.option("--quality", type=click.IntRange(1, 100), default=95, show_default=True)
@click.option(
    "--on-failure",
    type=click.Choice(["stop", "skip"]),
    default="stop",
    show_default=True,
    help="Фото, которое не прочиталось или уронило нормализацию: остановить сборку либо пропустить и перечислить в конце. Фото без ровно одной годной бутылки не теряется, оно идёт в коллекцию всем кадром",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=None,
    help="Первые N строк с фото: для пробы",
)
@click.option(
    "--report",
    "report_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="JSON с итогом и списком неудач",
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
    "--set",
    "overrides",
    multiple=True,
    metavar="секция.ключ=значение",
    help="Переопределить параметр конфига нормализации",
)
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

    Каждое фото проходит нормализацию, на нём должна найтись ровно одна годная бутылка, иначе с предупреждением берётся весь кадр как есть; по вектору на каждый вход encoder_input (crop, box_crop либо оба
    для гибридного поиска), в коллекции у каждого входа своё именованное пространство, а в хранилище картинок
    из конфига уходит вся бутылка по bbox маски, как box_crop у запроса, — её смотрит второй уровень; вектор — в коллекцию qdrant из конфига вместе с метаданными: slug, группа и все slug группы, попавшие в коллекцию, имя кропа,
    поля каталога, модель и её ревизия, отпечаток нормализации. Рядом с картинками пишется manifest.json: часть конфига нормализации,
    определяющая кропы, её отпечаток, коллекция и модель; поиск при старте сверяет с ним свой нормализатор.
    Устройства — device в конфигах, переменные VIS_SEARCHER_DEV и NORMALIZER_DEV их перекрывают.
    Дешёвые проверки идут до загрузки моделей: CSV и наличие всех фото, qdrant и отсутствие коллекции, доступность хранилища.
    """
    cfg = load_config(config_path)
    norm_cfg = load_norm_config(norm_config, list(overrides))
    normalization = crop_config(norm_cfg)
    fingerprint = normalization_fingerprint(normalization)
    rows, without_photo, has_group = read_rows(
        csv_path, slug_column, photo_column, group_column, limit
    )
    logger.info(
        f"{csv_path.name}: {len(rows)} строк с фото пойдут в коллекцию, {len(without_photo)} без фото пропускаются"
        + (f", например {without_photo[:3]}" if without_photo else "")
    )
    check_photos(rows, images)
    device = resolve_device(cfg.device)
    files = fetch_model(cfg.model, device.precision, cfg.revision)
    client = connect(cfg.qdrant)
    collection = cfg.qdrant.collection
    ensure_absent(client, collection)
    store = make_store(cfg.images, cfg.cache_dir)
    store.ensure_available()
    if store.exists(MANIFEST):
        previous = store.get_json(MANIFEST)
        if previous["fingerprint"] != fingerprint:
            logger.warning(
                f"в хранилище ({store.description}) лежат картинки коллекции {previous['collection']} с другой нормализацией ({previous['fingerprint']}): "
                f"они перепишутся, и та коллекция перестанет проходить проверку при старте"
            )
    norm = Normalizer(norm_cfg, norm_config.resolve().parent)
    logger.info(
        f"конфиг {config_path}: модель {cfg.model}, входы энкодера {'+'.join(cfg.encoder_input)}, коллекция {collection}; каталог: {len(rows)} строк с фото; "
        f"группы: {'колонка ' + group_column if has_group else 'нет, позиция = группа'}; кропы: {store.description}; нормализация на {norm.device}"
    )
    encoder = Encoder(files, device, cfg.batch_size, cfg.cache_dir)
    logger.info(f"энкодер: {encoder.description}")
    create_collection(client, collection, encoder.embed_dim, cfg.encoder_input)
    state = build(
        rows,
        images,
        norm,
        encoder,
        store,
        fmt,
        quality,
        skip_failures=on_failure == "skip",
        encoder_input=cfg.encoder_input,
        normalization=fingerprint,
    )
    store.put_json(
        MANIFEST,
        {
            "normalization": normalization,
            "fingerprint": fingerprint,
            "collection": collection,
            FIELD_MODEL: files.repo,
            FIELD_REVISION: files.revision,
            FIELD_ENCODER_INPUT: list(cfg.encoder_input),
            "images": {"crop": "bottle_box", "format": fmt, "quality": quality},
            "built_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        },
    )
    upsert(client, collection, state.points())
    report(state, rows, len(without_photo), has_group, collection, report_path)
    if cfg.snapshots is not None:
        logger.info(
            f"снапшот в хранилище снапшотов сам не обновляется; чтобы пустой qdrant поднимал эту сборку: "
            f"python -m vinishko.pred.pipeline.steps.vis_searcher.dump_collection --config {config_path}"
        )


if __name__ == "__main__":
    main()
