"""Оценка пайплайна на тестовом наборе: нормализация и поиск на каждом фото, recall@k по slug и качество отказа.

Запуск из корня: python -m vinishko.pipeline.steps.vis_searcher.evaluate --test-dir datasets/hack-vine/test
Разметка — CSV с колонками image_filename и slug; пустой slug значит, что ответа в каталоге нет и верный ответ — отказ.
"""

import csv
import json
import shutil
import statistics
import time
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from datetime import UTC
from datetime import datetime
from pathlib import Path

import rich_click as click
from kostyl.utils import setup_logger
from rich.console import Console
from rich.progress import BarColumn
from rich.progress import MofNCompleteColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from rich.table import Table

from vinishko.pipeline.pipeline import Pipeline
from vinishko.pipeline.pipeline import PipelineResult
from vinishko.pipeline.steps.normalization.normalize import HERE as NORM_DIR
from vinishko.pipeline.steps.normalization.normalize import Normalizer
from vinishko.pipeline.steps.normalization.normalize import (
    load_config as load_norm_config,
)
from vinishko.pipeline.steps.normalization.normalize import polys_box
from vinishko.pipeline.steps.vis_searcher.catalog import FIELD_GROUP_SLUGS
from vinishko.pipeline.steps.vis_searcher.catalog import FIELD_SLUG
from vinishko.pipeline.steps.vis_searcher.configs import DEFAULT_CONFIG
from vinishko.pipeline.steps.vis_searcher.configs import load_config
from vinishko.pipeline.steps.vis_searcher.search import VisSearcher
from vinishko.pipeline.structs import BottleCandidates
from vinishko.pipeline.structs import BottleCrop
from vinishko.pipeline.structs import RejectedBottle
from vinishko.pipeline.structs import UnmatchedBottle


KS = (1, 3, 5)
DUMPS = ("misses", "all", "none")
console = Console()
logger = setup_logger(fmt="detailed")


@dataclass(slots=True)
class Row:
    """Итог по одному тестовому фото."""

    image: str
    slug: str
    """Ожидаемый slug; пусто — ответа в каталоге нет, верный ответ — отказ."""
    in_catalog: bool
    """Ожидаемый slug есть в коллекции; иначе фото проверяет отказ."""
    status: str
    """candidates — кандидаты есть; no_match — поиск отказал; no_bottle — нормализация не нашла годной бутылки."""
    bottles: int
    """Сколько бутылок нашла нормализация, годных и нет."""
    detail: str
    """Причина отказа нормализации либо поиска."""
    top1: str
    top1_score: float | None
    rank: int | None
    """Позиция ожидаемого slug среди кандидатов бутылки с лучшим скором отбора, с 1."""
    rank_any: int | None
    """Лучшая позиция ожидаемого slug по всем бутылкам фото: для полок, где целевая бутылка не первая."""
    group_hit: bool
    """Ожидаемый slug состоит в группе какого-то кандидата первой бутылки: ответ на уровне группы верный."""
    seconds: float


def read_test(path: Path) -> list[tuple[str, str]]:
    """Пары (имя файла, ожидаемый slug); slug может быть пустым."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        for column in ("image_filename", "slug"):
            if column not in columns:
                raise click.UsageError(
                    f"в {path.name} нет колонки {column!r}; есть {columns}"
                )
        return [
            (rec["image_filename"].strip(), (rec["slug"] or "").strip())
            for rec in reader
        ]


def catalog_slugs(searcher: VisSearcher) -> set[str]:
    """Все slug коллекции, постранично."""
    slugs: set[str] = set()
    offset = None
    while True:
        points, offset = searcher.client.scroll(
            searcher.collection,
            limit=1024,
            offset=offset,
            with_payload=[FIELD_SLUG],
            with_vectors=False,
        )
        slugs.update(p.payload[FIELD_SLUG] for p in points if p.payload)
        if offset is None:
            return slugs


def rank_of(slug: str, answer: BottleCandidates | UnmatchedBottle) -> int | None:
    """Позиция slug среди кандидатов ответа, с 1."""
    if isinstance(answer, UnmatchedBottle):
        return None
    slugs = [c.slug for c in answer.candidates]
    return slugs.index(slug) + 1 if slug in slugs else None


def evaluate_image(
    pipeline: Pipeline, path: Path, slug: str, known: set[str]
) -> tuple[Row, PipelineResult]:
    """Прогон одного фото; ответ считается по бутылке с лучшим скором отбора, она в выдаче нормализации первая."""
    started = time.perf_counter()
    result = pipeline(path)
    seconds = time.perf_counter() - started
    in_catalog = bool(slug) and slug in known
    bottles = len(result.normalization)
    if not result.search:
        reasons = "; ".join(
            item.message
            for item in result.normalization
            if isinstance(item, RejectedBottle)
        )
        row = Row(
            path.name,
            slug,
            in_catalog,
            "no_bottle",
            bottles,
            reasons or "бутылок не найдено",
            "",
            None,
            None,
            None,
            False,
            seconds,
        )
        return row, result
    answer = result.search[0]
    ranks = [r for r in (rank_of(slug, a) for a in result.search) if r is not None]
    rank_any = min(ranks) if ranks else None
    if isinstance(answer, UnmatchedBottle):
        row = Row(
            path.name,
            slug,
            in_catalog,
            "no_match",
            bottles,
            answer.rejection.detail,
            "",
            None,
            None,
            rank_any,
            False,
            seconds,
        )
        return row, result
    top = answer.candidates[0]
    group_hit = any(
        slug in c.payload.get(FIELD_GROUP_SLUGS, []) for c in answer.candidates
    )
    row = Row(
        path.name,
        slug,
        in_catalog,
        "candidates",
        bottles,
        "",
        top.slug,
        top.score,
        rank_of(slug, answer),
        rank_any,
        group_hit,
        seconds,
    )
    return row, result


def share(flags: Iterable[bool]) -> float | None:
    """Доля истинных; None, если считать не по чему."""
    values = list(flags)
    return round(sum(values) / len(values), 4) if values else None


def quantiles(values: list[float]) -> dict[str, float] | None:
    """Медиана и края распределения косинуса."""
    if not values:
        return None
    values = sorted(values)
    return {
        "n": len(values),
        "min": round(values[0], 3),
        "p10": round(values[len(values) // 10], 3),
        "median": round(statistics.median(values), 3),
        "max": round(values[-1], 3),
    }


def metrics(rows: Sequence[Row]) -> dict:
    """Recall@k и качество отказа.

    with_answer — slug есть в коллекции, по ним recall. without_answer — slug пустой, по ним отказ: correct_reject и false_accept.
    answer_not_indexed — slug задан, но в коллекции его нет (фото каталога не прошло нормализацию либо позиции нет в CSV): такие фото
    ни в recall, ни в отказ не идут, для них отдельно not_indexed_accepted — доля, где поиск всё же выдал кандидатов, обычно соседей по группе.
    """
    with_answer = [r for r in rows if r.in_catalog]
    without = [r for r in rows if not r.slug]
    unindexed = [r for r in rows if r.slug and not r.in_catalog]
    out: dict = {
        "images": len(rows),
        "with_answer": len(with_answer),
        "without_answer": len(without),
        "answer_not_indexed": len(unindexed),
    }
    for k in KS:
        out[f"recall@{k}"] = share(
            r.rank is not None and r.rank <= k for r in with_answer
        )
    for k in KS:
        out[f"recall@{k}_any_bottle"] = share(
            r.rank_any is not None and r.rank_any <= k for r in with_answer
        )
    out["group_recall"] = share(r.group_hit for r in with_answer)
    out["false_reject"] = share(r.status == "no_match" for r in with_answer)
    out["correct_reject"] = share(r.status == "no_match" for r in without)
    out["false_accept"] = share(r.status == "candidates" for r in without)
    out["not_indexed_accepted"] = share(r.status == "candidates" for r in unindexed)
    out["no_bottle"] = share(r.status == "no_bottle" for r in rows)
    out["seconds_per_image"] = (
        round(statistics.mean(r.seconds for r in rows), 3) if rows else None
    )
    out["top1_cos"] = {
        "hits": quantiles(
            [
                r.top1_score
                for r in with_answer
                if r.rank == 1 and r.top1_score is not None
            ]
        ),
        "wrong": quantiles(
            [
                r.top1_score
                for r in with_answer
                if r.status == "candidates" and r.rank != 1 and r.top1_score is not None
            ]
        ),
        "false_accept": quantiles(
            [r.top1_score for r in without if r.top1_score is not None]
        ),
    }
    return out


def dump_image(
    out: Path, path: Path, result: PipelineResult, searcher: VisSearcher
) -> None:
    """Папка одного фото: копия исходника, normalization.json со всеми бутылками и причинами отказов, дамп поиска, если он был."""
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, out / f"source{path.suffix.lower()}")
    bottles = []
    for item in result.normalization:
        entry = {
            "uuid": item.uuid,
            "score": item.score,
            "bottle_box": polys_box(item.bottle),
        }
        if isinstance(item, BottleCrop):
            bottles.append({"status": "ok", "index": item.index, **entry})
        else:
            bottles.append(
                {
                    "status": "rejected",
                    "stage": item.rejection.stage,
                    "reason": item.rejection.reason,
                    "label": item.rejection.label,
                    "message": item.message,
                    **entry,
                }
            )
    (out / "normalization.json").write_text(
        json.dumps(
            {
                "bottles": bottles,
                "timings_s": {k: round(v, 3) for k, v in result.timings.items()},
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    if result.search:
        searcher.dump(result.search, out)


def is_miss(row: Row) -> bool:
    """Фото, на которое стоит посмотреть глазами: не нашли, нашли не то или приняли чужое."""
    return (row.in_catalog and row.rank != 1) or (
        not row.in_catalog and row.status == "candidates"
    )


def print_metrics(report: dict) -> None:
    """Сводка в терминал."""
    table = Table(
        title=f"{report['collection']} · {report['mode']} · top_k {report['top_k']}"
    )
    table.add_column("метрика")
    table.add_column("значение", justify="right")
    m = report["metrics"]
    for name in (
        "images",
        "with_answer",
        "without_answer",
        "answer_not_indexed",
        *(f"recall@{k}" for k in KS),
        "group_recall",
        "recall@1_any_bottle",
        "false_reject",
        "correct_reject",
        "false_accept",
        "not_indexed_accepted",
        "no_bottle",
        "seconds_per_image",
    ):
        value = m[name]
        table.add_row(
            name,
            "—"
            if value is None
            else (
                f"{value:.1%}"
                if isinstance(value, float) and name != "seconds_per_image"
                else str(value)
            ),
        )
    for name, q in m["top1_cos"].items():
        table.add_row(
            f"cos top-1, {name}",
            "—"
            if q is None
            else f"n={q['n']} min {q['min']} p10 {q['p10']} med {q['median']} max {q['max']}",
        )
    console.print(table)


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


@click.command()
@click.option(
    "--test-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("datasets/hack-vine/test"),
    show_default=True,
    help="Директория с test.csv и images/",
)
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Разметка вместо <test-dir>/test.csv",
)
@click.option(
    "--images",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Фото вместо <test-dir>/images",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_CONFIG,
    show_default=True,
    help="Конфиг визуального поиска; его debug_path не используется",
)
@click.option(
    "-c",
    "--norm-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=NORM_DIR / "normalize.toml",
    show_default=True,
)
@click.option(
    "--set",
    "overrides",
    multiple=True,
    metavar="секция.ключ=значение",
    help="Переопределить параметр конфига нормализации",
)
@click.option("--limit", type=click.IntRange(min=1), default=None, help="Первые N фото")
@click.option(
    "--dump",
    type=click.Choice(DUMPS),
    default="all",
    show_default=True,
    help="Папка на фото в <out>/dumps/<фото>: исходник, normalization.json с бутылками и причинами отказов, дамп поиска; все фото, только промахи либо ничего",
)
@click.option(
    "-o",
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Куда писать report.json, per_image.csv и дампы; по умолчанию reports/vis_searcher/<время>",
)
def main(
    test_dir: Path,
    csv_path: Path | None,
    images: Path | None,
    config_path: Path,
    norm_config: Path,
    overrides: tuple[str, ...],
    limit: int | None,
    dump: str,
    out: Path | None,
) -> None:
    """Прогнать тестовый набор через нормализацию и поиск, посчитать recall@k и качество отказа.

    Ответ по фото — кандидаты бутылки с лучшим скором отбора. recall@k считается по фото, чей slug есть в коллекции; фото с пустым slug
    проверяют отказ: correct_reject — поиск отказал, false_accept — выдал кандидатов. false_reject — ответ в каталоге был, а поиск отказал.
    """
    pairs = read_test(csv_path or test_dir / "test.csv")[:limit]
    images = images or test_dir / "images"
    missing = [name for name, _ in pairs if not (images / name).is_file()]
    if missing:
        raise click.UsageError(
            f"в {images} нет {len(missing)} фото из разметки, например {missing[:3]}"
        )
    out = out or Path("reports/vis_searcher") / datetime.now(
        tz=UTC
    ).astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)
    cfg.debug_path = None
    searcher = VisSearcher(cfg)
    known = catalog_slugs(searcher)
    pipeline = Pipeline(
        Normalizer(
            load_norm_config(norm_config, list(overrides)), norm_config.resolve().parent
        ),
        searcher,
    )
    logger.info(
        f"тест: {len(pairs)} фото, с ответом в коллекции {sum(slug in known for _, slug in pairs)}, без ответа {sum(not slug for _, slug in pairs)}; коллекция {searcher.info.count} позиций"
    )
    rows: list[Row] = []
    with progress() as bar:
        task = bar.add_task("тест", total=len(pairs))
        for name, slug in pairs:
            row, result = evaluate_image(pipeline, images / name, slug, known)
            rows.append(row)
            if dump == "all" or (dump == "misses" and is_miss(row)):
                dump_image(
                    out / "dumps" / Path(name).stem, images / name, result, searcher
                )
            bar.update(task, advance=1)
    report = {
        "collection": searcher.info.name,
        "model": f"{searcher.files.repo}@{searcher.files.revision[:8]}",
        "encoder_input": list(cfg.encoder_input),
        "mode": cfg.search.mode,
        "search": cfg.search.model_dump(),
        "top_k": cfg.top_k,
        "device": str(searcher.device),
        "metrics": metrics(rows),
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    with (out / "per_image.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(Row)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    print_metrics(report)
    logger.info(f"отчёт: {out}")


if __name__ == "__main__":
    main()
