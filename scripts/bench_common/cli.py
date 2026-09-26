import os
import re
from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import rich_click as click
import torch
from rich.console import Console
from rich.progress import BarColumn
from rich.progress import DownloadColumn
from rich.progress import MofNCompleteColumn
from rich.progress import Progress
from rich.progress import ProgressColumn
from rich.progress import SpinnerColumn
from rich.progress import Task
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from rich.progress import TransferSpeedColumn
from rich.table import Table
from rich.text import Text

from scripts.bench_common.data import Split
from scripts.bench_common.data import index_markup
from scripts.bench_common.data import limit_split
from scripts.bench_common.data import normalized
from scripts.bench_common.data import protocol_rows
from scripts.bench_common.data import read_split
from scripts.bench_common.metrics import ModeResult
from scripts.bench_common.report import column_title
from scripts.bench_common.report import ReportSpec
from scripts.bench_common.report import build_html
from scripts.bench_common.report import metric_sections
from vinishko.pipeline.steps.normalization.normalize import HERE as NORM_DIR

console = Console()

ROOT = Path(__file__).resolve().parents[2]
MODES = {"raw": ["raw"], "norm": ["norm"], "both": ["raw", "norm"]}
PROTOCOLS = {"loo": ["loo"], "oneshot": ["oneshot"], "both": ["loo", "oneshot"]}
SEED = 0

Dropped = dict[str, tuple[int, int]]
"""По ролям: сколько картинок нормализация не допустила до энкодера и сколько их было."""


# ---------- прогресс ----------


class RateColumn(ProgressColumn):
    """Скорость в штуках в секунду."""

    def render(self, task: Task) -> Text:
        """Текущая оценка скорости задачи."""
        return Text(
            f"{task.speed:.1f} шт/с" if task.speed else "— шт/с",
            style="progress.data.speed",
        )


def count_progress() -> Progress:
    """Бар по штукам со скоростью: эмбеддинги, поиск, примеры отчёта."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[name]}"),
        BarColumn(),
        MofNCompleteColumn(),
        RateColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def bytes_progress() -> Progress:
    """Бар по байтам: чтение разметки нормализации."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[name]}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


# ---------- параметры CLI ----------


def is_html(output: Path) -> bool:
    """Путь до файла отчёта, а не до директории."""
    return output.suffix.lower() == ".html"


def check_output(_ctx: click.Context, _param: click.Parameter, value: Path) -> Path:
    """Путь отчёта проверяется до прогона: опечатка не должна всплыть через час работы."""
    if not is_html(value) and value.is_file():
        raise click.BadParameter(
            f"{value} — файл без расширения .html: укажите директорию либо путь до .html"
        )
    (value.parent if is_html(value) else value).mkdir(parents=True, exist_ok=True)
    return value


def output_file(output: Path, finished: datetime) -> Path:
    """Путь с .html берётся как есть; иначе это директория, имя файла — время конца прогона."""
    return output if is_html(output) else output / f"{finished:%Y-%m-%d_%H-%M-%S}.html"


def parse_devices(
    _ctx: click.Context, _param: click.Parameter, value: str
) -> list[torch.device]:
    """Устройства из CLI: cpu либо список существующих видеокарт через запятую, cuda:0,cuda:1. CUDA здесь не поднимается: это дело воркеров."""
    names = [name.strip() for name in value.split(",")]
    if names == ["cpu"]:
        return [torch.device("cpu")]
    devices = []
    for name in names:
        match = re.fullmatch(r"cuda:(\d+)", name)
        if match is None:
            raise click.BadParameter(
                f"ожидается cpu либо видеокарты через запятую, например cuda:0 или cuda:0,cuda:1; получено {name!r}"
            )
        if not torch.cuda.is_available():
            raise click.BadParameter(
                "CUDA недоступна: проверьте драйвер либо укажите cpu"
            )
        index, count = int(match[1]), torch.cuda.device_count()
        if index >= count:
            raise click.BadParameter(
                f"видеокарты {index} нет: доступны индексы 0..{count - 1}"
            )
        devices.append(torch.device("cuda", index))
    if len(set(devices)) < len(devices):
        raise click.BadParameter(f"устройства повторяются: {value}")
    return devices


def common_options(command: Callable[..., Any]) -> Callable[..., Any]:
    """Параметры, одинаковые у замеров всех моделей: куда писать отчёт, на чём считать, какие данные и режимы."""
    datasets = ROOT / "datasets" / "visual-encoder"
    options = [
        click.option(
            "-o",
            "--output",
            required=True,
            type=click.Path(path_type=Path),
            callback=check_output,
            help="Куда положить html-отчёт: путь с .html используется как есть, иначе это директория, имя файла — время конца прогона",
        ),
        click.option(
            "--device",
            "devices",
            required=True,
            callback=parse_devices,
            help="cpu, cuda:<индекс> либо несколько видеокарт через запятую, cuda:0,cuda:1: на каждой своя копия модели, картинки делятся между ними поровну. bfloat16, если устройство считает в нём аппаратно, иначе float32",
        ),
        click.option(
            "--mode",
            type=click.Choice(list(MODES)),
            default="both",
            show_default=True,
            help="Вход энкодера: raw — целое фото, norm — кроп нормализации по normalization.jsonl, both — оба замера в одном отчёте",
        ),
        click.option(
            "--protocol",
            type=click.Choice(list(PROTOCOLS)),
            default="both",
            show_default=True,
            help="Что лежит в галерее: loo — вся val, запрос ищет среди всех остальных картинок, включая прочие фото своего класса; oneshot — одно фото на класс, как в каталоге, остальные фото класса идут запросами; both — оба, эмбеддинги общие",
        ),
        click.option(
            "--datasets-dir",
            type=click.Path(exists=True, file_okay=False, path_type=Path),
            default=datasets,
            show_default=True,
            help="Директория датасетов раскладки scripts/prepare_datasets.py, как data.datasets_dir в конфиге обучения: winesensed и products10k берутся из неё",
        ),
        click.option(
            "--winesensed",
            type=click.Path(exists=True, file_okay=False, path_type=Path),
            default=None,
            help="Переопределить путь до WineSensed; по умолчанию <datasets-dir>/winesensed",
        ),
        click.option(
            "--negatives",
            type=click.Path(exists=True, file_okay=False, path_type=Path),
            default=None,
            help="Переопределить датасет с negatives.json, запросами «не вино»; по умолчанию <datasets-dir>/products10k",
        ),
        click.option(
            "--distractors/--no-distractors",
            default=True,
            show_default=True,
            help="Добавить запросы val_distractors.json: вина, которых нет в галерее",
        ),
        click.option(
            "--norm-config",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            default=NORM_DIR / "normalize.toml",
            show_default=True,
            help="Конфиг нормализации: из него берутся параметры кропа и фона",
        ),
        click.option(
            "--workers",
            type=click.IntRange(min=0),
            default=min(8, os.cpu_count() or 1),
            show_default=True,
            help="Процессов даталоадера на каждое устройство: чтение, рендер кропа, предобработка",
        ),
        click.option(
            "--limit",
            type=click.IntRange(min=50),
            default=None,
            help="Не больше N картинок на роль, классы целиком; для пробного прогона",
        ),
        click.option(
            "--examples",
            type=click.IntRange(min=0),
            default=12,
            show_default=True,
            help="Примеров на раздел отчёта",
        ),
    ]
    for option in reversed(options):
        command = option(command)
    return command


# ---------- данные режима ----------


def dataset_paths(
    datasets_dir: Path, winesensed: Path | None, negatives: Path | None
) -> tuple[Path, Path]:
    """Директории WineSensed и негативов: явные пути имеют приоритет, иначе стандартная раскладка внутри --datasets-dir. Проверяется до прогона."""
    winesensed = winesensed or datasets_dir / "winesensed"
    negatives = negatives or datasets_dir / "products10k"
    for root, file in ((winesensed, "val.json"), (negatives, "negatives.json")):
        if not (root / "images").is_dir() or not (root / file).is_file():
            raise click.BadParameter(
                f"в {root} нет {file} либо images/: ожидается раскладка scripts/prepare_datasets.py",
                param_hint="--datasets-dir",
            )
    return winesensed, negatives


def read_splits(
    winesensed: Path, negatives: Path, distractors: bool, limit: int | None
) -> dict[str, Split]:
    """Сплиты замера по ролям: val всегда, distractors и negatives — запросы без ответа."""
    splits = {"val": read_split(winesensed, "val.json", "val")}
    if distractors:
        splits["distractors"] = read_split(
            winesensed, "val_distractors.json", "distractors"
        )
    splits["negatives"] = read_split(negatives, "negatives.json", "negatives")
    if limit is not None:
        splits = {role: limit_split(split, limit) for role, split in splits.items()}
    console.print(
        f"[green]data[/] {', '.join(f'{role}: {len(split)}' for role, split in splits.items())}"
    )
    return splits


def normalize_splits(splits: dict[str, Split]) -> dict[str, Split]:
    """Сплиты, в которых остались только картинки с годной бутылкой по normalization.jsonl своего датасета."""
    wanted: dict[Path, set[str]] = {}
    for split in splits.values():
        wanted.setdefault(split.root, set()).update(split.names)
    with bytes_progress() as progress:
        spans = {
            root: index_markup(root, names, progress) for root, names in wanted.items()
        }
    return {
        role: normalized(split, spans[split.root]) for role, split in splits.items()
    }


def prepare_mode(mode: str, raw: dict[str, Split]) -> tuple[dict[str, Split], Dropped]:
    """Сплиты режима и сколько картинок отсеяла нормализация. Роли, в которых не осталось картинок, выпадают; val остаться обязан."""
    splits = normalize_splits(raw) if mode == "norm" else raw
    dropped = (
        {role: (len(raw[role]) - len(splits[role]), len(raw[role])) for role in raw}
        if mode == "norm"
        else {}
    )
    splits = {role: split for role, split in splits.items() if len(split)}
    if "val" not in splits:
        raise click.ClickException("после нормализации в val не осталось картинок")
    if not len(protocol_rows(splits["val"], "loo")[1]):
        raise click.ClickException(
            "в val нет классов с двумя и более картинками: запросов не получается, увеличьте --limit"
        )
    return splits, dropped


# ---------- итог ----------


def metrics_table(results: list[ModeResult], spec: ReportSpec) -> Table:
    """Те же метрики, что в html: строка на метрику, столбец на режим."""
    table = Table(
        "метрика",
        *(column_title(r) for r in results),
        title=f"{spec.model} на WineSensed",
    )
    for column in table.columns[1:]:
        column.justify = "right"
    for section, rows in metric_sections(results, spec):
        table.add_section()
        table.add_row(f"[bold]{section}[/]", *[""] * len(results))
        for name, values in rows:
            table.add_row(name, *values)
    return table


def finish(
    results: list[ModeResult],
    spec: ReportSpec,
    run_info: dict[str, str],
    render_cfg: dict,
    examples: int,
    output: Path,
) -> None:
    """Таблица в консоль и html-отчёт на диск."""
    console.print(metrics_table(results, spec))
    finished = datetime.now(tz=UTC).astimezone()
    run_info = {**run_info, "конец прогона": f"{finished:%Y-%m-%d %H:%M:%S %Z}"}
    with count_progress() as progress:
        html = build_html(results, spec, run_info, render_cfg, examples, SEED, progress)
    path = output_file(output, finished)
    path.write_text(html, encoding="utf-8")
    console.print(f"[green]done[/] отчёт {path}, {path.stat().st_size / 1e6:.1f} МБ")
