import json
import os
import queue
import signal
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
from vinishko.pipeline.structs import Candidate, Rejection

console = Console()

SPLITS = ["val", "val_distractors", "train", "catalog", "negatives"]
OUT_NAME = "normalization.jsonl"
QUEUE_DEPTH = 4  # задач в очереди на воркера: хватает, чтобы никто не простаивал, и мало теряется при остановке


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


def record(name: str, items: list[Candidate | Rejection]) -> dict:
    """Строка normalization.jsonl: годные бутылки и отказы одной картинки, поля — как у Candidate и Rejection."""
    candidates = [asdict(item) for item in items if isinstance(item, Candidate)]
    return {
        "file": name,
        "status": "ok" if candidates else "no_bottles",
        "candidates": candidates,
        "rejections": [asdict(item) for item in items if isinstance(item, Rejection)],
    }


def error_record(name: str, error: Exception) -> dict:
    """Строка для картинки, на которой нормализатор упал: одна битая картинка не должна ронять прогон на сутки."""
    return {"file": name, "status": "error", "error": repr(error), "candidates": [], "rejections": []}


def worker(device: str, config: Path, overrides: list[str], tasks: Queue, results: Queue) -> None:
    """Процесс-воркер: свой Normalizer на своей видеокарте, берёт из tasks пары (путь, имя), кладёт в results строки разметки.

    Сообщения в results: ("ready", device) после загрузки модели, ("record", строка) на каждую картинку,
    ("fatal", текст) если модель не поднялась. None в tasks завершает воркер.
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
        except Exception as e:
            results.put(("record", error_record(name, e)))


def next_message(results: Queue, workers: list[BaseProcess]) -> tuple[str, Any]:
    """Следующее сообщение воркеров; если воркер умер молча, например его убил OOM-killer, прогон не зависает, а падает."""
    while True:
        try:
            return results.get(timeout=5)
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


def start_workers(devices: list[str], per_gpu: int, config: Path, overrides: list[str]) -> tuple[list[BaseProcess], Queue, Queue]:
    """Поднимает per_gpu воркеров на каждую видеокарту и ждёт, пока все загрузят модель."""
    ctx = get_context("spawn")  # fork после инициализации CUDA в родителе ломает драйвер, spawn даёт чистый процесс
    tasks, results = ctx.Queue(), ctx.Queue()
    workers: list[BaseProcess] = [
        ctx.Process(target=worker, args=(device, config, overrides, tasks, results), name=f"gpu{device}-{n}", daemon=True)
        for device in devices
        for n in range(per_gpu)
    ]
    for w in workers:
        w.start()
    for _ in workers:
        kind, payload = next_message(results, workers)
        if kind == "fatal":
            raise RuntimeError(payload)
    console.print(f"[green]ready[/] воркеров {len(workers)}: по {per_gpu} на GPU {','.join(devices)}")
    return workers, tasks, results


def annotate_dataset(
    dataset: Path, out: Path, meta: dict, splits: list[str] | None, limit: int | None, pool: tuple[list[BaseProcess], Queue, Queue], progress: Progress
) -> None:
    """Дописывает в out разметку картинок датасета, которых там ещё нет.

    Картинки раздаются воркерам по одной из общей очереди: куски не пересекаются, быстрая карта берёт больше.
    В очереди держится небольшой запас задач, файл пишет только этот процесс, порядок строк — по готовности.
    """
    workers, tasks, results = pool
    done = done_files(out)
    todo = [n for n in files_in_order(dataset, dataset_splits(dataset, splits)) if n not in done]
    if limit is not None:
        todo = todo[:limit]
    check_meta(out.with_suffix(".meta.json"), meta, len(done))
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    bar = progress.add_task(dataset.name, name=dataset.name, total=len(todo))
    counts = {"ok": 0, "no_bottles": 0, "error": 0}
    pending = iter(todo)
    in_flight = 0
    for name in islice(pending, QUEUE_DEPTH * len(workers)):
        tasks.put((str(dataset / "images" / name), name))
        in_flight += 1
    with out.open("a", encoding="utf-8") as f:
        while in_flight:
            _, rec = next_message(results, workers)
            in_flight -= 1
            counts[str(rec["status"])] += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()  # строка на диске сразу: прогон идёт сутками, и его могут убить в любой момент
            progress.update(bar, advance=1)
            for name in islice(pending, 1):
                tasks.put((str(dataset / "images" / name), name))
                in_flight += 1
    progress.remove_task(bar)
    console.print(f"[green]done[/] {dataset.name}: {len(done)} were done, +{len(todo)} now ({counts}) → {out}")

# !!!! Надо учесть при трейне, что в каталоге будут лежать изображения без фона, возможно вообще стоит отказаться от идеи с рамытием фона как входом и учить чисто без фона - но у нас тогда могут быть проблема с сегметацией

# И при eval нужно будет формировать бд так, что делаем поиск с размытым фоном, а в бд лежат изображения без, те, что будет на проде

@click.command()
@click.argument("datasets", nargs=-1, required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-c", "--config", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=NORM_DIR / "normalize.toml", show_default=True)
@click.option("--set", "overrides", multiple=True, metavar="секция.ключ=значение", help="Переопределить параметр конфига")
@click.option("--splits", default=None, help=f"Какие сплиты и в каком порядке, через запятую; по умолчанию те из {','.join(SPLITS)}, что есть в датасете")
@click.option("--limit", type=int, default=None, help="Не больше N новых картинок на датасет")
@click.option("--out", "out_name", default=OUT_NAME, show_default=True, help="Имя выходного файла внутри директории датасета либо абсолютный путь")
@click.option("--gpus", default=None, help="Номера видеокарт через запятую, по умолчанию все доступные; картинки делятся между их воркерами")
@click.option("--workers-per-gpu", type=click.IntRange(min=1), default=1, show_default=True, help="Процессов с моделью на одну видеокарту; SAM3 занимает около 5 ГБ видеопамяти на процесс")
def main(
    datasets: tuple[Path, ...], config: Path, overrides: tuple[str, ...], splits: str | None, limit: int | None, out_name: str, gpus: str | None, workers_per_gpu: int
) -> None:
    """Прогоняет Normalizer.annotate (SAM3 → отбор → проверка этикетки, без рендера) по картинкам развёрнутого датасета и пишет разметку в <dataset>/normalization.jsonl.

    Одна строка на картинку: status, candidates — годные бутылки с полями Candidate: index, score, bottle, label, angle, uuid;
    rejections — отказы с полями Rejection: reason, detail, score, bottle, label, uuid. Маски bottle и label — полигоны в пикселях оригинала после EXIF.
    Кропы не пишутся: их рендерит даталоадер по маскам и углу, с джиттером.
    Рядом пишется normalization.meta.json с конфигом прогона. Дозапись: уже обработанные картинки пропускаются, но только с теми же настройками,
    что в meta; порядок сплитов задаёт --splits. Запуск из корня: python -m scripts.normalize_dataset.

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
        for w in pool[0]:
            w.terminate()

if __name__ == "__main__":
    main()
