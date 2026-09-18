import asyncio
import hashlib
import json
import os
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Literal

import httpx
import pyarrow.parquet as pq
import rich_click as click
from huggingface_hub import HfApi, get_token, hf_hub_url
from huggingface_hub.hf_api import RepoFile
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

console = Console()

USER_AGENT = "wine-scanner-hack/0.1 (hackathon; contact via github)"
CONCURRENCY = 16
CHUNK = 1 << 20
BATCH_ROWS = 256
VAL_FRAC = 0.1
MIN_VAL_CLASS_SIZE = 2

OFF_API = "https://world.openfoodfacts.org/api/v2/search"
OFF_CATEGORY = "wines"
OFF_MAX_PRODUCTS = 20000
OFF_PAGE_SIZE = 100
OFF_IMAGE_KEYS = ("image_front_url", "image_packaging_url", "image_ingredients_url")
OFF_SEARCH_INTERVAL = 6.5  # поиск OFF лимитирован 10 запросами в минуту
OFF_LIST_RETRIES, OFF_LIST_SLEEP = 5, 10.0
OFF_IMAGE_RETRIES, OFF_IMAGE_SLEEP = 5, 30.0

Role = Literal["train", "catalog", "negatives"]
# какие json пишет источник каждой роли: имя файла → {имя картинки: метка}
ROLE_FILES: dict[Role, list[str]] = {
    "train": ["train", "val", "val_distractors"],
    "catalog": ["catalog"],
    "negatives": ["negatives"],
}

Item = tuple[str, str, bytes]  # имя файла, метка класса, байты картинки
Downloader = Callable[[httpx.AsyncClient, Path, Progress, bool], Awaitable[None]]
Unpacker = Callable[[Path, Progress, TaskID], Iterator[Item]]


@dataclass(frozen=True)
class Source:
    """Источник данных: как скачать в RAW/<name>, как развернуть и что написать о нём в README.

    Новый источник — это функция-распаковщик, выдающая (имя файла, метка, байты), и одна запись в SOURCES.
    Для репозитория HuggingFace загрузчик даёт hf_repo; для другого хостинга пишется свой Downloader.
    Роль определяет, какие json появятся рядом с images: см. ROLE_FILES.
    """

    name: str
    role: Role
    download: Downloader
    unpack: Unpacker
    origin: str
    content: str
    label: str
    filename: str
    license: str
    notes: str = ""


# ---------- прогресс ----------


def bytes_progress() -> Progress:
    """Бар по байтам для скачивания."""
    return Progress(
        TextColumn("{task.fields[name]}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def count_progress() -> Progress:
    """Бар по штукам: датасеты при скачивании, картинки при развороте."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[name]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    )


# ---------- скачивание: HuggingFace ----------


async def download_file(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    size: int | None,
    progress: Progress,
    dataset_task: TaskID,
    local: bool,
    name: str,
    headers: dict | None = None,
) -> None:
    """Потоковая загрузка с докачкой; в global-режиме двигает бар датасета, в local — свой бар."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    if size and existing >= size:
        if not local:
            progress.update(dataset_task, advance=size)
        return

    headers = dict(headers or {})
    if existing:
        headers["Range"] = f"bytes={existing}-"
    task = (
        progress.add_task("f", name=name, total=size, completed=existing)
        if local
        else dataset_task
    )
    if not local:
        progress.update(dataset_task, advance=existing)

    async with client.stream("GET", url, headers=headers) as r:
        if r.status_code == httpx.codes.REQUESTED_RANGE_NOT_SATISFIABLE:
            return
        r.raise_for_status()
        resumed = r.status_code == httpx.codes.PARTIAL_CONTENT
        if existing and not resumed:
            if local:
                progress.update(task, completed=0)
            else:
                progress.update(dataset_task, advance=-existing)
        with dest.open("ab" if resumed else "wb") as f:
            async for chunk in r.aiter_bytes(CHUNK):
                f.write(chunk)
                progress.update(task, advance=len(chunk))
    if local:
        progress.remove_task(task)


def hf_repo(repo_id: str, include: str = "*") -> Downloader:
    """Загрузчик файлов датасет-репозитория HuggingFace, чьи пути подходят под шаблон include (синтаксис fnmatch)."""

    async def download(
        client: httpx.AsyncClient, out: Path, progress: Progress, local: bool
    ) -> None:
        tree = HfApi().list_repo_tree(repo_id, repo_type="dataset", recursive=True)
        files = [
            f for f in tree if isinstance(f, RepoFile) and fnmatch(f.path, include)
        ]
        dataset_task = progress.add_task(
            "ds", name=repo_id, total=sum(f.size or 0 for f in files), visible=not local
        )
        token = get_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        sem = asyncio.Semaphore(CONCURRENCY)

        async def one(f: RepoFile) -> None:
            async with sem:
                url = hf_hub_url(repo_id, f.path, repo_type="dataset")
                await download_file(
                    client,
                    url,
                    out / f.path,
                    f.size,
                    progress,
                    dataset_task,
                    local,
                    f.path,
                    headers,
                )

        await asyncio.gather(*(one(f) for f in files))
        progress.remove_task(dataset_task)

    return download


# ---------- скачивание: Open Food Facts ----------


class GiveUpError(Exception):
    """Сервер не ответил за отведённое число попыток."""


async def get_with_retry(
    client: httpx.AsyncClient, url: str, retries: int, sleep: float, **kwargs
) -> httpx.Response:
    """GET с фиксированной паузой между попытками; после исчерпания — GiveUpError."""
    for attempt in range(retries):
        try:
            r = await client.get(url, **kwargs)
            if r.status_code < httpx.codes.BAD_REQUEST:
                return r
            status: object = r.status_code
        except httpx.TransportError as e:
            status = type(e).__name__
        console.print(
            f"[yellow]{status}[/] {url.split('?')[0]} — attempt {attempt + 1}/{retries}, sleep {sleep:.0f}s"
        )
        await asyncio.sleep(sleep)
    raise GiveUpError(url)


async def off_products(
    client: httpx.AsyncClient, start_page: int
) -> AsyncIterator[dict]:
    """Товары категории wines постранично, с паузой под лимит поиска."""
    page = start_page
    while True:
        r = await get_with_retry(
            client,
            OFF_API,
            OFF_LIST_RETRIES,
            OFF_LIST_SLEEP,
            params={
                "categories_tags_en": OFF_CATEGORY,
                "fields": "code,product_name,brands,categories_tags,"
                + ",".join(OFF_IMAGE_KEYS),
                "page_size": OFF_PAGE_SIZE,
                "page": page,
            },
        )
        products = r.json().get("products", [])
        if not products:
            return
        for p in products:
            yield p
        page += 1
        await asyncio.sleep(OFF_SEARCH_INTERVAL)


def off_jobs(products: Iterable[dict], img_dir: Path) -> list[tuple[str, Path]]:
    """Пары (url, путь) для всех фото товаров: images/<code>/{front,packaging,ingredients}.jpg."""
    jobs = []
    for p in products:
        for key in OFF_IMAGE_KEYS:
            url = p.get(key)
            if url:
                stem = key.removeprefix("image_").removesuffix("_url")
                jobs.append((url, img_dir / p["code"] / f"{stem}.jpg"))
    return jobs


async def off_listing(
    client: httpx.AsyncClient, out: Path, progress: Progress
) -> dict[str, dict]:
    """Товары в products.jsonl с продолжением со страницы обрыва; при отказе сервера возвращает собранное."""
    meta_path = out / "products.jsonl"
    seen: dict[str, dict] = {}
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                seen[p["code"]] = p

    start_page = len(seen) // OFF_PAGE_SIZE + 1
    listing = progress.add_task(
        "list",
        name=f"OFF: listing (from page {start_page})",
        total=OFF_MAX_PRODUCTS,
        completed=len(seen),
    )
    with meta_path.open("a", encoding="utf-8") as meta:
        try:
            async for p in off_products(client, start_page):
                if len(seen) >= OFF_MAX_PRODUCTS:
                    break
                if p["code"] in seen:
                    continue
                seen[p["code"]] = p
                meta.write(json.dumps(p, ensure_ascii=False) + "\n")
                progress.update(listing, advance=1)
        except GiveUpError:
            console.print(
                f"[yellow]OFF listing gave up[/] — {len(seen)} products collected, downloading their images"
            )
    progress.remove_task(listing)
    return seen


async def off_images(
    client: httpx.AsyncClient,
    jobs: list[tuple[str, Path]],
    progress: Progress,
    local: bool,
) -> None:
    """Качает фото; при отказе сервера этап завершается, остальное докачается следующим запуском."""
    dataset_task = progress.add_task(
        "ds", name=f"OFF images ({len(jobs)} files)", total=len(jobs), visible=not local
    )
    sem = asyncio.Semaphore(CONCURRENCY)
    stop = asyncio.Event()

    async def one(url: str, dest: Path) -> None:
        async with sem:
            if stop.is_set():
                return
            r = await get_with_retry(client, url, OFF_IMAGE_RETRIES, OFF_IMAGE_SLEEP)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            progress.update(dataset_task, advance=1)

    tasks = [asyncio.create_task(one(u, d)) for u, d in jobs]
    try:
        await asyncio.gather(*tasks)
    except GiveUpError:
        stop.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        done = sum(1 for _, d in jobs if d.exists())
        console.print(
            f"[yellow]OFF images gave up[/] — {done}/{len(jobs)} downloaded, rest on next run"
        )
    progress.remove_task(dataset_task)


async def download_off(
    client: httpx.AsyncClient, out: Path, progress: Progress, local: bool
) -> None:
    """Листинг товаров категории wines, затем фото тех из них, которых ещё нет на диске."""
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    products = await off_listing(client, out, progress)
    jobs = [(u, d) for u, d in off_jobs(products.values(), img_dir) if not d.exists()]
    await off_images(client, jobs, progress, local)


async def download_all(
    raw: Path, sources: list[Source], local: bool, proxy: str | None
) -> None:
    """Скачивает источники по очереди в raw/<name>."""
    overall = count_progress()
    transfer = bytes_progress()
    timeout = httpx.Timeout(60, read=600)
    with Live(Group(overall, transfer), console=console, refresh_per_second=8):
        overall_task = overall.add_task("all", name="datasets", total=len(sources))
        async with httpx.AsyncClient(
            proxy=proxy,
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            for source in sources:
                overall.update(overall_task, name=f"datasets · {source.name}")
                await source.download(client, raw / source.name, transfer, local)
                overall.update(overall_task, advance=1)
                console.print(
                    f"[green]downloaded[/] {source.name} → {raw / source.name}"
                )


# ---------- разворот ----------


def parquet_rows(files: list[Path]) -> int:
    """Число строк по метаданным parquet, без чтения данных."""
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


def iter_parquet(
    path: Path, columns: list[str], batch_rows: int = BATCH_ROWS
) -> Iterator[dict]:
    """Строки parquet-файла словарями, батчами, чтобы не держать шард в памяти."""
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
        yield from batch.to_pylist()


def unpack_winesensed(raw: Path, progress: Progress, task: TaskID) -> Iterator[Item]:
    """Метка — vintage_id: имя вина, год и winery_id заполнены лишь у 0.1% строк. Строки без картинки пропускаются."""
    files = sorted((raw / "data" / "wines").glob("*.parquet"))
    progress.update(task, total=parquet_rows(files))
    for f in files:
        for row in iter_parquet(f, ["image", "vintage_id"]):
            if row["image"] is None:
                continue
            yield row["image"]["path"], str(row["vintage_id"]), row["image"]["bytes"]


def unpack_products10k(raw: Path, progress: Progress, task: TaskID) -> Iterator[Item]:
    """Метка — sku; имя файла из img_file вида test/3963142.jpg."""
    files = sorted((raw / "data").glob("*.parquet"))
    progress.update(task, total=parquet_rows(files))
    for f in files:
        for row in iter_parquet(f, ["image", "img_file", "sku"]):
            yield row["img_file"].replace("/", "_"), row["sku"], row["image"]["bytes"]


def unpack_off(raw: Path, progress: Progress, task: TaskID) -> Iterator[Item]:
    """Метка — штрихкод; в каталог идёт только фронтальное фото, одно на товар."""
    files = sorted((raw / "images").glob("*/front.jpg"))
    progress.update(task, total=len(files))
    for f in files:
        yield f"{f.parent.name}.jpg", f.parent.name, f.read_bytes()


def is_val(label: str, val_frac: float) -> bool:
    """Детерминированный сплит по классу: все картинки класса попадают в одну часть."""
    h = int.from_bytes(hashlib.blake2b(label.encode(), digest_size=4).digest(), "big")
    return h / (1 << 32) < val_frac


def split_labels(
    role: Role, labels: dict[str, str], val_frac: float
) -> dict[str, dict[str, str]]:
    """Раскладывает {картинка: метка} по json-файлам роли; сплит по классам нужен только роли train."""
    if role != "train":
        return {ROLE_FILES[role][0]: labels}
    class_size = Counter(labels.values())
    splits: dict[str, dict[str, str]] = {name: {} for name in ROLE_FILES[role]}
    for filename, label in labels.items():
        if not is_val(label, val_frac):
            split = "train"
        elif class_size[label] >= MIN_VAL_CLASS_SIZE:
            split = "val"
        else:
            split = "val_distractors"
        splits[split][filename] = label
    return splits


def write_image(dest: Path, data: bytes) -> None:
    """Пишет байты как есть; уже записанный файл того же размера не трогает."""
    if dest.exists() and dest.stat().st_size == len(data):
        return
    dest.write_bytes(data)


def unpack_source(
    source: Source,
    raw: Path,
    out: Path,
    progress: Progress,
    limit: int | None,
    val_frac: float,
) -> None:
    """Пишет картинки в out/images и json-файлы роли источника в out."""
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    task = progress.add_task(source.name, name=source.name, total=None)
    labels: dict[str, str] = {}
    conflicts = 0
    for filename, label, data in source.unpack(raw, progress, task):
        if limit is not None and len(labels) >= limit:
            break
        progress.update(task, advance=1)
        if filename in labels:
            conflicts += labels[filename] != label
            continue
        labels[filename] = label
        write_image(img_dir / filename, data)
    progress.remove_task(task)
    parts = split_labels(source.role, labels, val_frac)
    for name, part in parts.items():
        with (out / f"{name}.json").open("w", encoding="utf-8") as f:
            json.dump(part, f, ensure_ascii=False, indent=0)
    stats = ", ".join(
        f"{name} {len(part)} images / {len(set(part.values()))} classes"
        for name, part in parts.items()
    )
    console.print(
        f"[green]unpacked[/] {source.name}: {stats}, {conflicts} label conflicts → {out}"
    )


README = """\
# Внешние датасеты для обучения и проверки визуального энкодера

Скачано и развёрнуто скриптом `scripts/prepare_datasets.py`. Картинки лежат байт в байт как в исходных дампах, без перекодирования.

## Раскладка

```
<dataset>/
  images/          все картинки датасета, плоская папка
  <файлы роли>     json вида {{"имя файла": "метка класса"}}
```

Метки — строки, уникальные только внутри датасета. При склейке датасетов метку нужно префиксовать именем датасета.

## Роли и их файлы

- **train** — обучение и оценка латентного пространства: `train.json`, `val.json`, `val_distractors.json`.
- **catalog** — пример каталога, одно фото на товар: `catalog.json`. Идёт в галерею как есть; своих запросов у него нет.
- **negatives** — запросы «не вино»: `negatives.json`. Правильный ответ на любой из них — «не найдено»; вместе с `val_distractors.json` калибруют порог отказа, но проверяют другой случай: на фото вообще не вино.

## Сплит роли train

Сплит по классам, а не по картинкам: все картинки класса попадают в одну часть, классы train и val не пересекаются. Класс попадает в val-долю детерминированно, по хэшу метки (blake2b), доля `--val-frac` = {val_frac}. Внутри val-доли:

- `train.json` — классы для обучения, включая одиночные (одна картинка на класс).
- `val.json` — классы с ≥{min_val} картинками. Одновременно галерея и запросы, схема leave-one-out: каждая картинка ищет среди всех остальных картинок val.
- `val_distractors.json` — одиночные классы из val-доли. Запросы, у которых в галерее нет ни одного позитива; правильный ответ на них — «не найдено». Нужны для калибровки порога отказа: скор top-1 и отрыв top-1 от top-2 на val против дистракторов.

Классы трёх файлов не пересекаются. Протокол оценки фиксирован и не меняется между итерациями нормализации и обучения.

## Датасеты

| Датасет | Роль | Источник | Что на картинках | Метка класса | Имя файла | Лицензия |
|---|---|---|---|---|---|---|
{sources}

{notes}

## Состояние на диске

{stats}
"""


def load_parts(source: Source, out: Path) -> dict[str, dict[str, str]] | None:
    """Json-файлы роли источника из out/<name>, либо None, если хотя бы одного нет."""
    parts = {}
    for name in ROLE_FILES[source.role]:
        path = out / source.name / f"{name}.json"
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as f:
            parts[name] = json.load(f)
    return parts


def write_readme(out: Path, val_frac: float) -> None:
    """README с описанием раскладки, зарегистрированных источников и счётчиками по тем из них, что уже развёрнуты в out."""
    sources = [
        f"| {s.name} | {s.role} | {s.origin} | {s.content} | {s.label} | {s.filename} | {s.license} |"
        for s in SOURCES.values()
    ]
    notes = [f"- **{s.name}**: {s.notes}" for s in SOURCES.values() if s.notes]
    stats = ["| Датасет | Файл | Картинок | Классов |", "|---|---|---|---|"]
    for source in SOURCES.values():
        for name, part in (load_parts(source, out) or {}).items():
            stats.append(
                f"| {source.name} | `{name}.json` | {len(part)} | {len(set(part.values()))} |"
            )
    text = README.format(
        val_frac=val_frac,
        min_val=MIN_VAL_CLASS_SIZE,
        sources="\n".join(sources),
        notes="\n".join(notes),
        stats="\n".join(stats),
    )
    (out / "README.md").write_text(text, encoding="utf-8")


def unpack_all(
    raw: Path, out: Path, sources: list[Source], limit: int | None, val_frac: float
) -> None:
    """Разворачивает источники из raw/<name> в out/<name> и обновляет out/README.md."""
    with count_progress() as progress:
        for source in sources:
            unpack_source(
                source, raw / source.name, out / source.name, progress, limit, val_frac
            )
    write_readme(out, val_frac)


# ---------- состояние на диске ----------


@dataclass(frozen=True)
class State:
    """Что есть на диске у одного источника."""

    expected: int  # картинок по json-файлам роли; 0, если json нет
    missing: int  # из них нет в images
    raw_bytes: int  # размер сырого дампа, 0 если его нет

    @property
    def unpacked(self) -> bool:
        """Развёрнут целиком: json на месте и все перечисленные в них картинки лежат в images."""
        return self.expected > 0 and self.missing == 0


def state_of(source: Source, raw: Path, out: Path) -> State:
    """Сверяет json-файлы роли с содержимым images и меряет сырой дамп. В сеть не ходит."""
    parts = load_parts(source, out)
    expected = {name for part in (parts or {}).values() for name in part}
    img_dir = out / source.name / "images"
    on_disk = {p.name for p in img_dir.iterdir()} if img_dir.is_dir() else set()
    raw_dir = raw / source.name
    raw_bytes = (
        sum(p.stat().st_size for p in raw_dir.rglob("*") if p.is_file())
        if raw_dir.is_dir()
        else 0
    )
    return State(len(expected), len(expected - on_disk), raw_bytes)


def human_size(n: int) -> str:
    """Размер в МБ или ГБ."""
    return f"{n / 2**30:.1f} ГБ" if n >= 2**30 else f"{n / 2**20:.0f} МБ"


def action_of(state: State) -> str:
    """Что sync сделает с источником."""
    if state.unpacked:
        return "ничего"
    return "докачать и развернуть" if state.raw_bytes else "скачать и развернуть"


def print_status(raw: Path, out: Path, sources: list[Source]) -> dict[str, State]:
    """Таблица состояния источников; возвращает состояния по именам."""
    table = Table(title=f"{out}  ·  raw: {raw}")
    for column in (
        "источник",
        "роль",
        "развёрнуто",
        "не хватает",
        "raw",
        "sync сделает",
    ):
        table.add_column(column)
    states = {}
    for source in sources:
        state = states[source.name] = state_of(source, raw, out)
        raw_size = human_size(state.raw_bytes) if state.raw_bytes else "нет"
        table.add_row(
            source.name,
            source.role,
            str(state.expected) if state.expected else "нет",
            str(state.missing) if state.expected else "—",
            raw_size,
            action_of(state),
        )
    console.print(table)
    return states


# ---------- реестр источников ----------

SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        Source(
            name="winesensed",
            role="train",
            download=hf_repo("christopher/winesensed", "data/wines/*"),
            unpack=unpack_winesensed,
            origin="HF `christopher/winesensed`, config `wines`",
            content="Пользовательские фото винных этикеток с Vivino, 480×640, бутылка обрезана сверху и снизу",
            label="`vintage_id` — винтаж Vivino (вино + год); id вина без винтажа в дампе нет",
            filename="Имя картинки из дампа",
            license="CC BY-NC-ND 4.0 — только прототип",
            notes="около 4.5 тыс. строк без картинки пропускаются; конфиги `napping` и `participants` не скачиваются.",
        ),
        Source(
            name="off",
            role="catalog",
            download=download_off,
            unpack=unpack_off,
            origin="Open Food Facts, поиск по категории `wines`",
            content="Фронтальные фото упаковок с телефона, одно на товар; заметная часть не вино (уксус, вермут, соки)",
            label="Штрихкод",
            filename="`<штрихкод>.jpg`",
            license="ODbL, фото CC BY-SA",
            notes="поиск OFF лимитирован и обрывается отказом сервера; каждый запуск продолжает листинг со страницы обрыва, "
            "поэтому размер каталога растёт от запуска к запуску. Название, бренд и `categories_tags` лежат в raw `products.jsonl`.",
        ),
        Source(
            name="products10k",
            role="negatives",
            download=hf_repo(
                "nyris/products10k-traintest-v1", "data/test-0000[05]-of-00009.parquet"
            ),
            unpack=unpack_products10k,
            origin="HF `nyris/products10k-traintest-v1`, два шарда test из девяти",
            content="Товары JD.com: одежда, сумки, техника, бытовая химия; вина нет, бутылок около 8% (косметика, соусы)",
            label="`sku`, для роли не важен",
            filename="`test_<n>.jpg` из `img_file`",
            license="research-only",
            notes="берётся кусок около 1 ГБ; бутылки не-вина в нём — самые трудные негативы.",
        ),
    ]
}


# ---------- CLI ----------

datasets_option = click.option(
    "--datasets",
    "-d",
    multiple=True,
    type=click.Choice(list(SOURCES)),
    help="Какие источники обрабатывать; по умолчанию все",
)
pb_option = click.option(
    "--pb",
    type=click.Choice(["global", "local"]),
    default="global",
    show_default=True,
    help="Бар на датасет или бар на каждый файл",
)
proxy_option = click.option(
    "--proxy", default=None, help="URL прокси, например http://127.0.0.1:2080"
)
limit_option = click.option(
    "--limit",
    type=int,
    default=None,
    help="Не больше N картинок на датасет, для проверки",
)
val_frac_option = click.option(
    "--val-frac",
    type=float,
    default=VAL_FRAC,
    show_default=True,
    help="Доля классов в val-доле у источников роли train; классы train и val не пересекаются",
)
raw_option = click.option(
    "--raw",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Где лежат сырые дампы; по умолчанию соседняя с OUT директория <OUT>-raw",
)


def selected(datasets: tuple[str, ...]) -> list[Source]:
    """Источники по именам из -d, либо все зарегистрированные."""
    return [SOURCES[name] for name in datasets or SOURCES]


def raw_for(out: Path, raw: Path | None) -> Path:
    """Директория сырых дампов: явная либо <OUT>-raw рядом с OUT."""
    return raw or out.with_name(f"{out.name}-raw")


def run_download(raw: Path, sources: list[Source], pb: str, proxy: str | None) -> None:
    """Общая часть команд download и sync."""
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
    raw.mkdir(parents=True, exist_ok=True)
    asyncio.run(download_all(raw, sources, pb == "local", proxy))


@click.group()
def cli() -> None:
    """Внешние датасеты для энкодера: скачать сырые дампы и развернуть их в картинки с метками."""


@cli.command()
@click.argument("out", type=click.Path(file_okay=False, path_type=Path))
@raw_option
@datasets_option
def status(out: Path, raw: Path | None, datasets: tuple[str, ...]) -> None:
    """Показывает, какие источники развёрнуты в OUT целиком, чего не хватает и что сделает sync. В сеть не ходит."""
    print_status(raw_for(out, raw), out, selected(datasets))


@cli.command()
@click.argument("out", type=click.Path(file_okay=False, path_type=Path))
@raw_option
@datasets_option
@pb_option
@proxy_option
@val_frac_option
@click.option(
    "--force",
    is_flag=True,
    help="Скачать и развернуть заново даже то, что уже развёрнуто целиком",
)
def sync(
    out: Path,
    raw: Path | None,
    datasets: tuple[str, ...],
    pb: str,
    proxy: str | None,
    val_frac: float,
    force: bool,
) -> None:
    """Доводит OUT до полного состояния: докачивает и разворачивает только те источники, которых не хватает.

    Источник считается готовым, если его json-файлы на месте и все перечисленные в них картинки лежат в images.
    Готовый источник не трогается, даже если его сырой дамп удалён: ради него ничего заново не качается.
    """
    raw = raw_for(out, raw)
    states = print_status(raw, out, selected(datasets))
    todo = [s for s in selected(datasets) if force or not states[s.name].unpacked]
    if not todo:
        console.print("[green]всё на месте[/]")
        return
    run_download(raw, todo, pb, proxy)
    unpack_all(raw, out, todo, None, val_frac)
    print_status(raw, out, selected(datasets))


@cli.command()
@click.argument("raw", type=click.Path(file_okay=False, path_type=Path))
@datasets_option
@pb_option
@proxy_option
def download(raw: Path, datasets: tuple[str, ...], pb: str, proxy: str | None) -> None:
    """Только скачивание сырых дампов в RAW/<dataset>, с докачкой."""
    run_download(raw, selected(datasets), pb, proxy)


@cli.command()
@click.argument("raw", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("out", type=click.Path(file_okay=False, path_type=Path))
@datasets_option
@limit_option
@val_frac_option
def unpack(
    raw: Path, out: Path, datasets: tuple[str, ...], limit: int | None, val_frac: float
) -> None:
    """Только разворот RAW/<dataset> в OUT/<dataset>/images и json-файлы роли."""
    unpack_all(raw, out, selected(datasets), limit, val_frac)


if __name__ == "__main__":
    cli()
