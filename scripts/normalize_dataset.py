import json
import os
import queue
import signal
import time
from collections.abc import Iterator
from dataclasses import asdict
from itertools import islice
from multiprocessing import get_context
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any

import rich_click as click
import torch
from PIL import Image
from rich.console import Console
from rich.table import Table
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from vinishko.pipeline.steps.normalization.normalize import HERE as NORM_DIR
from vinishko.pipeline.steps.normalization.normalize import Normalizer, load_config
from vinishko.pipeline.structs import BottleCrop, RejectedBottle

console = Console()

SPLITS = ["val", "val_distractors", "train", "catalog", "negatives"]
OUT_NAME = "normalization.jsonl"
QUEUE_DEPTH = 4  # задач в очереди на воркера: хватает, чтобы никто не простаивал, и мало теряется при остановке
BENCH_TOLERANCE = 0.03  # bench советует наименьшее число воркеров, чья скорость не хуже лучшей больше чем на эту долю

Pool = tuple[list[BaseProcess], Queue, Queue]


def count_progress() -> Progress:
    """Бар по картинкам датасета."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[name]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def dataset_splits(dataset: Path, splits: list[str] | None) -> list[str]:
    """Сплиты датасета в порядке обработки: заданные явно либо те из SPLITS, чьи json есть в директории."""
    if splits is not None:
        return splits
    found = [s for s in SPLITS if (dataset / f"{s}.json").exists()]
    if not found:
        raise click.UsageError(f"в {dataset} нет ни одного из {[f'{s}.json' for s in SPLITS]}: это директория, развёрнутая prepare_datasets?")
    return found


def files_in_order(dataset: Path, splits: list[str]) -> list[str]:
    """Имена картинок из json сплитов в заданном порядке, без повторов."""
    names: dict[str, None] = {}
    for split in splits:
        with (dataset / f"{split}.json").open(encoding="utf-8") as f:
            names.update(dict.fromkeys(json.load(f)))
    return list(names)


def drop_broken_tail(out: Path) -> None:
    """Обрезает незаконченную последнюю строку jsonl: она остаётся, если прошлый прогон убили посреди записи."""
    data = out.read_bytes()
    if data and not data.endswith(b"\n"):
        out.write_bytes(data[: data.rfind(b"\n") + 1])


def done_files(out: Path) -> set[str]:
    """Имена уже записанных в jsonl картинок, для дозаписи."""
    if not out.exists():
        return set()
    drop_broken_tail(out)
    with out.open(encoding="utf-8") as f:
        return {json.loads(line)["file"] for line in f if line.strip()}


def check_meta(meta_path: Path, meta: dict, n_done: int) -> None:
    """Дозапись допустима только с теми же настройками: иначе в одном файле смешается разметка разных версий."""
    if n_done and meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != meta:
        raise click.UsageError(
            f"{meta_path.name} записан с другими настройками нормализации, а в разметке уже {n_done} картинок: "
            "укажите другой --out либо удалите старую разметку"
        )


def record(name: str, items: list[BottleCrop | RejectedBottle]) -> dict:
    """Строка normalization.jsonl: годные бутылки и отказы одной картинки; у годных разметка без кропа, см. BottleCrop.markup."""
    candidates = [item.markup() for item in items if isinstance(item, BottleCrop)]
    return {
        "file": name,
        "status": "ok" if candidates else "no_bottles",
        "candidates": candidates,
        "rejections": [asdict(item) for item in items if isinstance(item, RejectedBottle)],
    }


def error_record(name: str, error: Exception) -> dict:
    """Строка для картинки, на которой нормализатор упал: одна битая картинка не должна ронять прогон на сутки."""
    return {"file": name, "status": "error", "error": repr(error), "candidates": [], "rejections": []}


def worker(device: str, config: Path, overrides: list[str], tasks: Queue, results: Queue) -> None:
    """Процесс-воркер: свой Normalizer на своей видеокарте, берёт из tasks пары (путь, имя), кладёт в results строки разметки.

    Сообщения в results: ("ready", device) после загрузки модели, ("record", строка) на каждую картинку,
    ("fatal", текст) если модель не поднялась либо посреди работы кончилась видеопамять. None в tasks завершает воркер.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C обрабатывает главный процесс и сам гасит воркеры
    os.environ["CUDA_VISIBLE_DEVICES"] = device  # CUDA поднимается лениво, до первого обращения видимость ещё можно задать
    try:
        norm = Normalizer(load_config(config, overrides), config.resolve().parent)
        norm.seg.query(Image.new("RGB", (64, 64)), [norm.seg.prompt])  # модель грузится на первом кадре: делаем это до очереди
    except Exception as e:  # иначе нехватка памяти при загрузке пометила бы ошибкой каждую картинку датасета
        results.put(("fatal", f"воркер на GPU {device}: {e!r}"))
        return
    results.put(("ready", device))
    while (task := tasks.get()) is not None:
        path, name = task
        try:
            results.put(("record", record(name, norm.annotate(path))))
        except torch.OutOfMemoryError as e:  # виновата не картинка, а число воркеров: прогон падает, картинка остаётся неразмеченной для дозаписи
            results.put(("fatal", f"воркер на GPU {device}: {e!r}"))
            return
        except Exception as e:
            results.put(("record", error_record(name, e)))


def next_message(results: Queue, workers: list[BaseProcess]) -> tuple[str, Any]:
    """Следующее сообщение воркеров. Прогон падает, если воркер прислал fatal либо умер молча, например его убил OOM-killer."""
    while True:
        try:
            kind, payload = results.get(timeout=5)
            if kind == "fatal":
                raise RuntimeError(payload)
            return kind, payload
        except queue.Empty:
            dead = [f"{w.name}: код {w.exitcode}" for w in workers if not w.is_alive()]
            if dead:
                raise RuntimeError(f"воркеры завершились: {dead}") from None


def all_gpus() -> list[str]:
    """Номера всех доступных видеокарт; если задан CUDA_VISIBLE_DEVICES, берутся карты из него."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return visible.split(",")
    count = torch.cuda.device_count()  # считает через NVML и не поднимает контекст CUDA в главном процессе
    if not count:
        raise click.UsageError("видеокарт не найдено: проверьте драйвер либо задайте --gpus явно")
    return [str(n) for n in range(count)]


def stop_workers(workers: list[BaseProcess]) -> None:
    """Гасит воркеры и ждёт их выхода: видеопамять освобождается только со смертью процесса."""
    for w in workers:
        w.terminate()
    for w in workers:
        w.join()


def start_workers(devices: list[str], per_gpu: int, config: Path, overrides: list[str]) -> Pool:
    """Поднимает per_gpu воркеров на каждую видеокарту и ждёт, пока все загрузят модель; если хоть один не поднялся, гасит остальных."""
    ctx = get_context("spawn")  # fork после инициализации CUDA в родителе ломает драйвер, spawn даёт чистый процесс
    tasks, results = ctx.Queue(), ctx.Queue()
    workers: list[BaseProcess] = [
        ctx.Process(target=worker, args=(device, config, overrides, tasks, results), name=f"gpu{device}-{n}", daemon=True)
        for device in devices
        for n in range(per_gpu)
    ]
    for w in workers:
        w.start()
    try:
        for _ in workers:
            next_message(results, workers)
    except BaseException:
        stop_workers(workers)
        raise
    console.print(f"[green]ready[/] воркеров {len(workers)}: по {per_gpu} на GPU {','.join(devices)}")
    return workers, tasks, results


def annotate_files(pool: Pool, dataset: Path, names: list[str]) -> Iterator[dict]:
    """Строки разметки картинок датасета, по готовности.

    Картинки раздаются воркерам по одной из общей очереди: куски не пересекаются, быстрая карта берёт больше.
    В очереди держится небольшой запас задач.
    """
    workers, tasks, results = pool
    pending = iter(names)
    in_flight = 0
    for name in islice(pending, QUEUE_DEPTH * len(workers)):
        tasks.put((str(dataset / "images" / name), name))
        in_flight += 1
    while in_flight:
        _, rec = next_message(results, workers)
        in_flight -= 1
        for name in islice(pending, 1):
            tasks.put((str(dataset / "images" / name), name))
            in_flight += 1
        yield rec


def annotate_dataset(dataset: Path, out: Path, meta: dict, splits: list[str] | None, limit: int | None, pool: Pool, progress: Progress) -> None:
    """Дописывает в out разметку картинок датасета, которых там ещё нет. Файл пишет только этот процесс, порядок строк — по готовности."""
    done = done_files(out)
    todo = [n for n in files_in_order(dataset, dataset_splits(dataset, splits)) if n not in done]
    if limit is not None:
        todo = todo[:limit]
    check_meta(out.with_suffix(".meta.json"), meta, len(done))
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    bar = progress.add_task(dataset.name, name=dataset.name, total=len(todo))
    counts = {"ok": 0, "no_bottles": 0, "error": 0}
    with out.open("a", encoding="utf-8") as f:
        for rec in annotate_files(pool, dataset, todo):
            counts[str(rec["status"])] += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()  # строка на диске сразу: прогон идёт сутками, и его могут убить в любой момент
            progress.update(bar, advance=1)
    progress.remove_task(bar)
    console.print(f"[green]done[/] {dataset.name}: {len(done)} were done, +{len(todo)} now ({counts}) → {out}")

# !!!! Надо учесть при трейне, что в каталоге будут лежать изображения без фона, возможно вообще стоит отказаться от идеи с рамытием фона как входом и учить чисто без фона - но у нас тогда могут быть проблема с сегметацией

# И при eval нужно будет формировать бд так, что делаем поиск с размытым фоном, а в бд лежат изображения без, те, что будет на проде

@click.group()
def cli() -> None:
    """Разметка нормализации по развёрнутому датасету: run размечает, bench подбирает число воркеров на видеокарту. Запуск из корня: python -m scripts.normalize_dataset."""


@cli.command()
@click.argument("datasets", nargs=-1, required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-c", "--config", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=NORM_DIR / "normalize.toml", show_default=True)
@click.option("--set", "overrides", multiple=True, metavar="секция.ключ=значение", help="Переопределить параметр конфига")
@click.option("--splits", default=None, help=f"Какие сплиты и в каком порядке, через запятую; по умолчанию те из {','.join(SPLITS)}, что есть в датасете")
@click.option("--limit", type=int, default=None, help="Не больше N новых картинок на датасет")
@click.option("--out", "out_name", default=OUT_NAME, show_default=True, help="Имя выходного файла внутри директории датасета либо абсолютный путь")
@click.option("--gpus", default=None, help="Номера видеокарт через запятую, по умолчанию все доступные; картинки делятся между их воркерами")
@click.option("--workers-per-gpu", type=click.IntRange(min=1), default=1, show_default=True, help="Процессов с моделью на одну видеокарту; SAM3 занимает около 5 ГБ видеопамяти на процесс, число подбирает bench")
def run(
    datasets: tuple[Path, ...], config: Path, overrides: tuple[str, ...], splits: str | None, limit: int | None, out_name: str, gpus: str | None, workers_per_gpu: int
) -> None:
    """Прогоняет Normalizer.annotate (SAM3 → отбор → проверка этикетки → кроп, который в разметку не пишется) по картинкам развёрнутого датасета и пишет разметку в <dataset>/normalization.jsonl.

    Одна строка на картинку: status, candidates — годные бутылки с полями BottleCrop.markup: index, score, bottle, label, angle, uuid;
    rejections — отказы с полями RejectedBottle: reason, detail, score, bottle, label, uuid. Маски bottle и label — полигоны в пикселях оригинала после EXIF.
    Кропы не пишутся: их рендерит даталоадер по маскам и углу, с джиттером.
    Рядом пишется normalization.meta.json с конфигом прогона. Дозапись: уже обработанные картинки пропускаются, но только с теми же настройками,
    что в meta; порядок сплитов задаёт --splits.

    Параллельность по данным: на каждой видеокарте, по умолчанию на всех, а с --gpus 0,2 на указанных, поднимается --workers-per-gpu процессов с моделью,
    картинки раздаются им из общей очереди без пересечений, файл пишет главный процесс.
    """
    norm = Normalizer(load_config(config, list(overrides)), config.resolve().parent)  # модель здесь не грузится, нужны тег и порог для meta
    meta = {"segmentation": norm.seg.tag, "threshold": round(norm.threshold, 4), "config": norm.cfg}
    pool = start_workers(gpus.split(",") if gpus else all_gpus(), workers_per_gpu, config, list(overrides))
    try:
        with count_progress() as progress:
            for dataset in datasets:
                out = Path(out_name) if Path(out_name).is_absolute() else dataset / out_name
                annotate_dataset(dataset, out, meta, splits.split(",") if splits else None, limit, pool, progress)
    finally:
        stop_workers(pool[0])


def bench_table(speeds: dict[int, float], n_gpus: int) -> Table:
    """Таблица замера: скорость при каждом числе воркеров на видеокарту и прирост к одному воркеру."""
    table = Table("воркеров на GPU", "всего воркеров", "с на фото", "фото в час", "к одному воркеру")
    for n, speed in speeds.items():
        table.add_row(str(n), str(n * n_gpus), f"{1 / speed:.3f}", f"{speed * 3600:,.0f}".replace(",", " "), f"{speed / speeds[1]:.2f}×")
    return table


@cli.command()
@click.argument("dataset", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-c", "--config", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=NORM_DIR / "normalize.toml", show_default=True)
@click.option("--set", "overrides", multiple=True, metavar="секция.ключ=значение", help="Переопределить параметр конфига; задавайте те же, что пойдут в run")
@click.option("--gpus", default=None, help="Номера видеокарт через запятую, по умолчанию все доступные")
@click.option("--max-workers", type=click.IntRange(min=1), default=4, show_default=True, help="До скольких воркеров на видеокарту пробовать")
@click.option("--limit", type=click.IntRange(min=1), default=200, show_default=True, help="Картинок на один замер")
def bench(dataset: Path, config: Path, overrides: tuple[str, ...], gpus: str | None, max_workers: int, limit: int) -> None:
    """Подбирает --workers-per-gpu: размечает одни и те же картинки датасета с 1, 2, … воркерами на видеокарту и сравнивает скорость.

    Разметка никуда не пишется. Время загрузки модели в замер не входит. Перебор останавливается раньше --max-workers, если очередной воркер
    не поместился в видеопамять при загрузке либо посреди замера; такой замер не засчитывается. Советуется наименьшее число воркеров, чья скорость не хуже лучшей больше чем на 3%:
    лишний воркер занимает около 5 ГБ видеопамяти и ядро процессора.
    """
    devices = gpus.split(",") if gpus else all_gpus()
    names = files_in_order(dataset, dataset_splits(dataset, None))[:limit]
    for name in names:
        (dataset / "images" / name).read_bytes()  # файлы в кэше страниц до первого замера: иначе чтение с диска достанется только ему
    speeds: dict[int, float] = {}
    for n in range(1, max_workers + 1):
        try:
            pool = start_workers(devices, n, config, list(overrides))
        except RuntimeError as e:
            console.print(f"[yellow]stop[/] {n} на GPU не поднялись: {e}")
            break
        try:
            with count_progress() as progress:
                bar = progress.add_task(str(n), name=f"{n} на GPU", total=len(names))
                started = time.perf_counter()
                errors = 0
                for rec in annotate_files(pool, dataset, names):
                    errors += rec["status"] == "error"
                    progress.update(bar, advance=1)
                elapsed = time.perf_counter() - started
        except RuntimeError as e:
            console.print(f"[yellow]stop[/] {n} на GPU упали посреди замера: {e}")
            break
        finally:
            stop_workers(pool[0])
        if errors:
            console.print(f"[yellow]stop[/] {errors} картинок с ошибкой при {n} на GPU: замер не засчитан, ошибки смотрите через run --limit")
            break
        speeds[n] = len(names) / elapsed
    if not speeds:
        raise click.ClickException("не поднялся ни один воркер")
    console.print(bench_table(speeds, len(devices)))
    best = min(n for n, speed in speeds.items() if speed >= (1 - BENCH_TOLERANCE) * max(speeds.values()))
    console.print(f"[green]best[/] --workers-per-gpu {best}: {speeds[best] * 3600:,.0f} фото в час на {len(devices)} GPU".replace(",", " "))


if __name__ == "__main__":
    cli()
