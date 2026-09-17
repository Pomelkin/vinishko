import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path

import pyarrow.parquet as pq
import rich_click as click
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

console = Console()

DATASETS = ["winesensed", "products10k", "sop", "rp2k"]
ALL_DATASETS = [*DATASETS, "off"]
MAGIC = {b"\xff\xd8\xff": "jpg", b"\x89PNG": "png", b"RIFF": "webp", b"GIF8": "gif"}
BATCH_ROWS = 256
VAL_FRAC = 0.1
MIN_VAL_CLASS_SIZE = 2
WINESENSED_SKIPPED = 4478
SPLITS = ["train", "val", "val_distractors"]
QRELS_BATCH_ROWS = 1 << 20

README = """\
# Внешние датасеты для обучения визуального энкодера

Развёрнуто скриптом `scripts/download_datasets.py` → `scripts/unpack_datasets.py`. Картинки лежат байт в байт как в исходных дампах, без перекодирования.

## Раскладка

```
<dataset>/
  images/               все картинки датасета, плоская папка
  train.json            {{"имя файла": "метка класса"}}
  val.json
  val_distractors.json
```

Метки — строки, уникальные только внутри датасета. При склейке датасетов метку нужно префиксовать именем датасета: `sku` из Products-10K и `vintage_id` из WineSensed — числовые строки и совпадают по значениям.

## Сплит

Сплит по классам, а не по картинкам: все картинки класса попадают в одну часть, классы train и val не пересекаются. Класс попадает в val-долю детерминированно, по хэшу метки (blake2b), доля `--val-frac` = {val_frac}. Внутри val-доли:

- `train.json` — классы для обучения, включая одиночные (одна картинка на класс).
- `val.json` — классы с ≥{min_val} картинками. Одновременно галерея и запросы, схема leave-one-out: каждая картинка ищет среди всех остальных картинок val.
- `val_distractors.json` — одиночные классы из val-доли. Запросы, у которых в галерее нет ни одного позитива; правильный ответ на них — «не найдено». Нужны для калибровки порога отказа: скор top-1 и отрыв top-1 от top-2 на val против дистракторов.

Классы трёх файлов не пересекаются. Протокол оценки фиксирован и не меняется между итерациями нормализации и обучения.

## Датасеты

| Датасет | Источник | Что на картинках | Метка класса | Имя файла | Лицензия |
|---|---|---|---|---|---|
| winesensed | HF `christopher/winesensed`, config `wines` | Пользовательские фото винных этикеток с Vivino | `vintage_id` — винтаж Vivino (вино + год). Имя вина, год и winery_id заполнены лишь у 0.1% строк, id вина без винтажа из дампа не получить | Имя картинки из дампа | CC BY-NC-ND 4.0 — только прототип |
| products10k | HF `nyris/products10k-traintest-v1` (Products-10K, JD.com) | Фото товаров, студийные и пользовательские | `sku` — товар; train и test оригинала слиты (одно пространство SKU) и заново разбиты по классам | `train_<n>.jpg` / `test_<n>.jpg` из `img_file` | research-only |
| sop | HF `JamieSJS/stanford-online-products` | Фото товаров с eBay | id товара — компонента связности по qrels, совпадает с префиксом id до `_` | id из corpus | CC BY 4.0 |
| rp2k | HF `JamieSJS/rp2k`, бенчмарк-срез, не полные 500k | Фото товаров с полок китайского ритейла | Компонента связности по qrels, представитель — минимальный id компоненты; классы тоньше названия товара | id из corpus: название товара + суффикс | Заявлено CC BY 4.0, проверить |
| off | Open Food Facts, категория wines | Фронтальные фото упаковок, одно на штрихкод, половина не вино | Штрихкод | `<code>_front.jpg` | ODbL — по умолчанию не разворачивается |

В WineSensed {ws_skipped} строк без картинки пропущены. Query-файлы SOP и RP2K не разворачиваются: они побайтно дублируют corpus.

## Состояние на диске

{stats}
"""

Item = tuple[str, str, bytes]
Unpacker = Callable[[Path, Progress, int], Iterator[Item]]


def count_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[name]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def ext_of(data: bytes) -> str:
    for magic, ext in MAGIC.items():
        if data.startswith(magic):
            return ext
    raise ValueError(f"unknown image format: {data[:8]!r}")


def parquet_rows(files: list[Path]) -> int:
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


def iter_parquet(path: Path, columns: list[str], batch_rows: int = BATCH_ROWS) -> Iterator[dict]:
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
        yield from batch.to_pylist()


class UnionFind:
    """Компоненты связности; меткой компоненты служит минимальный id в ней."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        parent = self.parent
        root = x
        while parent.setdefault(root, root) != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def labels(self) -> dict[str, str]:
        rep: dict[str, str] = {}
        for x in self.parent:
            root = self.find(x)
            rep[root] = min(rep.get(root, x), x)
        return {x: rep[self.find(x)] for x in self.parent}


def qrels_labels(files: list[Path]) -> dict[str, str]:
    uf = UnionFind()
    for f in files:
        for row in iter_parquet(f, ["query-id", "corpus-id", "score"], QRELS_BATCH_ROWS):
            if row["score"] > 0:
                uf.union(row["query-id"], row["corpus-id"])
    return uf.labels()


def unpack_winesensed(raw: Path, progress: Progress, task: int) -> Iterator[Item]:
    """Метка — vintage_id: имя вина, год и winery_id заполнены лишь у 0.1% строк."""
    files = sorted((raw / "data" / "wines").glob("*.parquet"))
    progress.update(task, total=parquet_rows(files))
    for f in files:
        for row in iter_parquet(f, ["image", "vintage_id"]):
            if row["image"] is None:
                continue
            yield row["image"]["path"], str(row["vintage_id"]), row["image"]["bytes"]


def unpack_products10k(raw: Path, progress: Progress, task: int) -> Iterator[Item]:
    """Метка — sku; img_file вида train/1.jpg или test/3963142.jpg, train и test делят одно пространство SKU."""
    files = sorted((raw / "data").glob("*.parquet"))
    progress.update(task, total=parquet_rows(files))
    for f in files:
        for row in iter_parquet(f, ["image", "img_file", "sku"]):
            yield row["img_file"].replace("/", "_"), row["sku"], row["image"]["bytes"]


def unpack_mteb(raw: Path, progress: Progress, task: int) -> Iterator[Item]:
    """SOP и RP2K: картинки из corpus*, метка — компонента связности по qrel*; query — копии corpus."""
    corpus = sorted(raw.glob("corpus*.parquet"))
    progress.update(task, total=parquet_rows(corpus))
    labels = qrels_labels(sorted(raw.glob("qrel*.parquet")))
    for f in corpus:
        for row in iter_parquet(f, ["id", "image"]):
            data = row["image"]
            yield f"{row['id']}.{ext_of(data)}", labels.get(row["id"], row["id"]), data


def unpack_off(raw: Path, progress: Progress, task: int) -> Iterator[Item]:
    """Метка — штрихкод (имя директории), файл — images/{code}/{front,packaging,ingredients}.jpg."""
    files = sorted((raw / "images").glob("*/*"))
    progress.update(task, total=len(files))
    for f in files:
        yield f"{f.parent.name}_{f.name}", f.parent.name, f.read_bytes()


UNPACKERS: dict[str, Unpacker] = {
    "winesensed": unpack_winesensed,
    "products10k": unpack_products10k,
    "sop": unpack_mteb,
    "rp2k": unpack_mteb,
    "off": unpack_off,
}


def write_image(dest: Path, data: bytes) -> None:
    if dest.exists() and dest.stat().st_size == len(data):
        return
    dest.write_bytes(data)


def is_val(label: str, val_frac: float) -> bool:
    """Детерминированный сплит по классу: все картинки класса попадают в одну часть."""
    h = int.from_bytes(hashlib.blake2b(label.encode(), digest_size=4).digest(), "big")
    return h / (1 << 32) < val_frac


def unpack(name: str, raw: Path, out: Path, progress: Progress, limit: int | None, val_frac: float) -> None:
    """Пишет картинки в out/images и сплит по классам: train.json, val.json (классы с ≥2 картинками, галерея и запросы) и val_distractors.json (одиночные классы — запросы без позитива, для порога отказа)."""
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    task = progress.add_task(name, name=name, total=None)
    labels: dict[str, str] = {}
    conflicts = 0
    for filename, label, data in UNPACKERS[name](raw, progress, task):
        if limit is not None and len(labels) >= limit:
            break
        progress.update(task, advance=1)
        if filename in labels:
            conflicts += labels[filename] != label
            continue
        labels[filename] = label
        write_image(img_dir / filename, data)
    progress.remove_task(task)
    class_size = Counter(labels.values())
    splits: dict[str, dict[str, str]] = {split: {} for split in SPLITS}
    for filename, label in labels.items():
        if not is_val(label, val_frac):
            split = "train"
        elif class_size[label] >= MIN_VAL_CLASS_SIZE:
            split = "val"
        else:
            split = "val_distractors"
        splits[split][filename] = label
    for split, part in splits.items():
        with (out / f"{split}.json").open("w", encoding="utf-8") as f:
            json.dump(part, f, ensure_ascii=False, indent=0)
    stats = ", ".join(f"{split} {len(part)} images / {len(set(part.values()))} classes" for split, part in splits.items())
    console.print(f"[green]done[/] {name}: {stats}, {conflicts} label conflicts → {out}")


def write_readme(out: Path, val_frac: float) -> None:
    """README с описанием раскладки и таблицей по датасетам, чьи json уже лежат в out."""
    rows = ["| Датасет | train | val | val_distractors |", "|---|---|---|---|"]
    for d in sorted(p.parent for p in out.glob("*/train.json")):
        cells = []
        for split in SPLITS:
            with (d / f"{split}.json").open(encoding="utf-8") as f:
                part = json.load(f)
            cells.append(f"{len(part)} картинок / {len(set(part.values()))} классов")
        rows.append(f"| {d.name} | " + " | ".join(cells) + " |")
    text = README.format(val_frac=val_frac, min_val=MIN_VAL_CLASS_SIZE, ws_skipped=WINESENSED_SKIPPED, stats="\n".join(rows))
    (out / "README.md").write_text(text, encoding="utf-8")


@click.command()
@click.argument("raw", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("out", type=click.Path(path_type=Path))
@click.option("--datasets", "-d", multiple=True, type=click.Choice(ALL_DATASETS), help="Какие датасеты разворачивать; по умолчанию все, кроме off (одно фото на штрихкод, половина не вино)")
@click.option("--limit", type=int, default=None, help="Не больше N картинок на датасет, для проверки")
@click.option("--val-frac", type=float, default=VAL_FRAC, show_default=True, help="Доля классов в val среди классов с ≥2 картинками; классы train и val не пересекаются")
def main(raw: Path, out: Path, datasets: tuple[str, ...], limit: int | None, val_frac: float) -> None:
    """Разворачивает результат download_datasets.py из RAW в OUT/<dataset>/images/ и OUT/<dataset>/{train,val,val_distractors}.json (имя файла → метка класса)."""
    with count_progress() as progress:
        for name in datasets or DATASETS:
            unpack(name, raw / name, out / name, progress, limit, val_frac)
    write_readme(out, val_frac)


if __name__ == "__main__":
    main()
