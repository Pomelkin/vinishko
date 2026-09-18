import json
from dataclasses import asdict
from pathlib import Path

import rich_click as click
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

SPLITS = ["val", "val_distractors", "train"]
OUT_NAME = "normalization.jsonl"


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


def files_in_order(dataset: Path, splits: list[str]) -> list[str]:
    """Имена картинок из json сплитов в заданном порядке, без повторов."""
    names: dict[str, None] = {}
    for split in splits:
        with (dataset / f"{split}.json").open(encoding="utf-8") as f:
            names.update(dict.fromkeys(json.load(f)))
    return list(names)


def done_files(out: Path) -> set[str]:
    """Имена уже записанных в jsonl картинок, для дозаписи."""
    if not out.exists():
        return set()
    with out.open(encoding="utf-8") as f:
        return {json.loads(line)["file"] for line in f if line.strip()}


def record(name: str, items: list[Candidate | Rejection]) -> dict:
    """Строка normalization.jsonl: годные бутылки и отказы одной картинки, поля — как у Candidate и Rejection."""
    candidates = [asdict(item) for item in items if isinstance(item, Candidate)]
    return {
        "file": name,
        "status": "ok" if candidates else "no_bottles",
        "candidates": candidates,
        "rejections": [asdict(item) for item in items if isinstance(item, Rejection)],
    }


def annotate_dataset(dataset: Path, out: Path, norm: Normalizer, splits: list[str], limit: int | None, progress: Progress) -> None:
    """Дописывает в out разметку картинок датасета, которых там ещё нет."""
    done = done_files(out)
    todo = [n for n in files_in_order(dataset, splits) if n not in done]
    if limit is not None:
        todo = todo[:limit]
    task = progress.add_task(dataset.name, name=dataset.name, total=len(todo))
    counts = {"ok": 0, "no_bottles": 0, "error": 0}
    meta = {"segmentation": norm.seg.tag, "threshold": round(norm.threshold, 4), "config": norm.cfg}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    with out.open("a", encoding="utf-8") as f:
        for name in todo:
            rec: dict
            try:
                rec = record(name, norm.annotate(dataset / "images" / name))
            except Exception as e:  # одна битая картинка не должна ронять прогон на сутки
                rec = {"file": name, "status": "error", "error": repr(e), "candidates": [], "rejections": []}
            counts[str(rec["status"])] += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            progress.update(task, advance=1)
    progress.remove_task(task)
    console.print(f"[green]done[/] {dataset.name}: {len(done)} were done, +{len(todo)} now ({counts}) → {out}")

# !!!! Надо учесть при трейне, что в каталоге будут лежать изображения без фона, возможно вообще стоит отказаться от идеи с рамытием фона как входом и учить чисто без фона - но у нас тогда могут быть проблема с сегметацией

# И при eval нужно будет формировать бд так, что делаем поиск с размытым фоном, а в бд лежат изображения без, те, что будет на проде

@click.command()
@click.argument("datasets", nargs=-1, required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-c", "--config", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=NORM_DIR / "normalize.toml", show_default=True)
@click.option("--set", "overrides", multiple=True, metavar="секция.ключ=значение", help="Переопределить параметр конфига")
@click.option("--splits", default=",".join(SPLITS), show_default=True, help="Какие сплиты и в каком порядке")
@click.option("--limit", type=int, default=None, help="Не больше N новых картинок на датасет")
@click.option("--out", "out_name", default=OUT_NAME, show_default=True, help="Имя выходного файла внутри директории датасета либо абсолютный путь")
def main(datasets: tuple[Path, ...], config: Path, overrides: tuple[str, ...], splits: str, limit: int | None, out_name: str) -> None:
    """Прогоняет Normalizer.annotate (SAM3 → отбор → проверка этикетки, без рендера) по картинкам развёрнутого датасета и пишет разметку в <dataset>/normalization.jsonl.

    Одна строка на картинку: status, candidates — годные бутылки с полями Candidate: index, score, bottle, label, angle, uuid;
    rejections — отказы с полями Rejection: reason, detail, score, bottle, label, uuid. Маски bottle и label — полигоны в пикселях оригинала после EXIF.
    Кропы не пишутся: их рендерит даталоадер по маскам и углу, с джиттером.
    Дозапись: уже обработанные картинки пропускаются, порядок сплитов задаёт --splits. Запуск из корня: python -m scripts.normalize_dataset.
    """
    cfg = load_config(config, list(overrides))
    norm = Normalizer(cfg, config.resolve().parent)
    with count_progress() as progress:
        for dataset in datasets:
            out = Path(out_name) if Path(out_name).is_absolute() else dataset / out_name
            annotate_dataset(dataset, out, norm, splits.split(","), limit, progress)


if __name__ == "__main__":
    main()
