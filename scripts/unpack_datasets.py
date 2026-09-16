import hashlib
import json
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
QRELS_BATCH_ROWS = 1 << 20

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
    """Пишет картинки в out/images и сплит по классам в out/{train,val}.json."""
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
    splits: dict[str, dict[str, str]] = {"train": {}, "val": {}}
    for filename, label in labels.items():
        splits["val" if is_val(label, val_frac) else "train"][filename] = label
    for split, part in splits.items():
        with (out / f"{split}.json").open("w", encoding="utf-8") as f:
            json.dump(part, f, ensure_ascii=False, indent=0)
    stats = ", ".join(f"{split} {len(part)} images / {len(set(part.values()))} classes" for split, part in splits.items())
    console.print(f"[green]done[/] {name}: {stats}, {conflicts} label conflicts → {out}")


@click.command()
@click.argument("raw", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("out", type=click.Path(path_type=Path))
@click.option("--datasets", "-d", multiple=True, type=click.Choice(ALL_DATASETS), help="Какие датасеты разворачивать; по умолчанию все, кроме off (одно фото на штрихкод, половина не вино)")
@click.option("--limit", type=int, default=None, help="Не больше N картинок на датасет, для проверки")
@click.option("--val-frac", type=float, default=VAL_FRAC, show_default=True, help="Доля классов в val; классы train и val не пересекаются")
def main(raw: Path, out: Path, datasets: tuple[str, ...], limit: int | None, val_frac: float) -> None:
    """Разворачивает результат download_datasets.py из RAW в OUT/<dataset>/images/ и OUT/<dataset>/{train,val}.json (имя файла → метка класса)."""
    with count_progress() as progress:
        for name in datasets or DATASETS:
            unpack(name, raw / name, out / name, progress, limit, val_frac)


if __name__ == "__main__":
    main()
