"""Сквозная проверка через API: фото тестового набора уходят в POST /recognize, ответы сравниваются с разметкой, метрики классификации.

Промежуточные картинки не сохраняются, это проверка работоспособности и качества сервиса как чёрного ящика; разбор шагов с дампами —
vinishko/pipeline/evaluate_pipeline.py. Запуск из корня при поднятом сервисе: python -m vinishko.e2e --url http://127.0.0.1:8000
Разметка — CSV с колонками image_filename и slug; пустой slug значит, что верный ответ — отказ.
"""

import csv
import json
import statistics
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from datetime import UTC
from datetime import datetime
from pathlib import Path

import httpx
import rich_click as click
from kostyl.utils import setup_logger
from rich.console import Console
from rich.table import Table

from vinishko.pipeline.steps.vis_searcher.evaluate import progress
from vinishko.pipeline.steps.vis_searcher.evaluate import read_test
from vinishko.pipeline.steps.vis_searcher.evaluate import share


console = Console()
logger = setup_logger(fmt="detailed")


@dataclass(slots=True)
class Row:
    """Ответ API по одному фото против разметки."""

    image: str
    slug: str
    """Ожидаемый slug; пусто — верный ответ отказ."""
    http_status: int
    bottles: int
    """Сколько бутылок вернул сервис."""
    status: str
    """Исход первой бутылки по скору отбора: matched, rejected, candidates; no_bottle — бутылок нет; error — сервис не ответил."""
    answer: str
    """slug выбранной позиции первой бутылки; пусто, если отказ."""
    stage: str
    """Шаг отказа первой бутылки; пусто, если ответ есть."""
    reason: str
    source: str
    correct: bool
    """Верный slug либо отказ на фото без ответа; ошибка сервиса — неверно."""
    seconds: float


def ask(client: httpx.Client, path: Path) -> tuple[int, dict | None, float]:
    """POST /recognize: код ответа, JSON и время."""
    started = time.perf_counter()
    with path.open("rb") as file:
        response = client.post("/recognize", files={"image": (path.name, file)})
    seconds = time.perf_counter() - started
    body = (
        response.json()
        if response.headers.get("content-type", "").startswith("application/json")
        else None
    )
    return response.status_code, body, seconds


def row_of(
    name: str, slug: str, status_code: int, body: dict | None, seconds: float
) -> Row:
    """Строка отчёта по ответу сервиса."""
    if status_code != 200 or body is None:
        return Row(name, slug, status_code, 0, "error", "", "", "", "", False, seconds)
    bottles = body["bottles"]
    if not bottles:
        return Row(
            name, slug, status_code, 0, "no_bottle", "", "", "", "", not slug, seconds
        )
    first = bottles[0]
    match, rejection = first.get("match"), first.get("rejection")
    answer = match["slug"] if match else ""
    correct = answer == slug if slug else not match
    return Row(
        name,
        slug,
        status_code,
        len(bottles),
        first["status"],
        answer,
        rejection["stage"] if rejection else "",
        rejection["reason"] if rejection else "",
        match["source"] if match else "",
        correct,
        seconds,
    )


def metrics(rows: list[Row]) -> dict:
    """Классификация итогового ответа по всем фото: система выдаёт один slug либо отказ."""
    with_answer = [r for r in rows if r.slug]
    without = [r for r in rows if not r.slug]
    answered = [r for r in rows if r.answer]
    correct_answers = sum(r.correct for r in with_answer)
    precision = correct_answers / len(answered) if answered else None
    recall = correct_answers / len(with_answer) if with_answer else None
    seconds = sorted(r.seconds for r in rows if r.status != "error")
    return {
        "images": len(rows),
        "errors": sum(r.status == "error" for r in rows),
        "accuracy": share(r.correct for r in rows),
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(2 * precision * recall / (precision + recall), 4)
        if precision and recall
        else None,
        "confusion": {
            "answer_correct": correct_answers,
            "answer_wrong": sum(bool(r.answer) and not r.correct for r in with_answer),
            "answer_on_empty": sum(bool(r.answer) for r in without),
            "reject_with_answer": sum(not r.answer for r in with_answer),
            "reject_on_empty": sum(not r.answer for r in without),
        },
        "rejected_by_stage": {
            stage: sum(r.stage == stage for r in rows)
            for stage in ("normalization", "search", "resolve")
        },
        "no_bottle": sum(r.status == "no_bottle" for r in rows),
        "resolved_by_model": sum(
            r.source in {"ndr_v5", "filter_v1"} or r.stage == "resolve" for r in rows
        ),
        "seconds": {
            "mean": round(statistics.mean(seconds), 3),
            "p50": round(statistics.median(seconds), 3),
            "p90": round(seconds[int(0.9 * (len(seconds) - 1))], 3),
        }
        if seconds
        else None,
    }


def print_report(report: dict) -> None:
    """Сводка в терминал."""
    health = report["health"]
    table = Table(
        title=f"{report['url']} · коллекция {health.get('collection')} · второй уровень {health.get('resolver') or 'нет'}"
    )
    table.add_column("метрика")
    table.add_column("значение", justify="right")
    for name, value in report["metrics"].items():
        table.add_row(name, format_metric(name, value))
    console.print(table)


def format_metric(name: str, value: object) -> str:
    """Значение метрики строкой: доли в процентах, словари через запятую."""
    if value is None:
        return "—"
    if isinstance(value, dict):
        return ", ".join(f"{k} {v}" for k, v in value.items())
    if isinstance(value, float) and name in ("accuracy", "precision", "recall", "f1"):
        return f"{value:.1%}"
    return str(value)


@click.command()
@click.option(
    "--url", default="http://127.0.0.1:8000", show_default=True, help="Адрес сервиса"
)
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
    "--limit", type=click.IntRange(min=1), default=None, help="Первые N фото разметки"
)
@click.option(
    "--timeout",
    type=float,
    default=300.0,
    show_default=True,
    help="Секунд на один запрос",
)
@click.option(
    "-o",
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Куда писать report.json и per_image.csv; по умолчанию reports/e2e/<время>",
)
def main(
    url: str,
    test_dir: Path,
    csv_path: Path | None,
    images: Path | None,
    limit: int | None,
    timeout: float,
    out: Path | None,
) -> None:
    """Проверить сервис на тестовом наборе: /health, потом каждое фото в /recognize, метрики классификации итогового ответа."""
    pairs = read_test(csv_path or test_dir / "test.csv")[:limit]
    images = images or test_dir / "images"
    missing = [name for name, _ in pairs if not (images / name).is_file()]
    if missing:
        raise click.UsageError(
            f"в {images} нет {len(missing)} фото из разметки, например {missing[:3]}"
        )
    out = out or Path("reports/e2e") / datetime.now(tz=UTC).astimezone().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    with httpx.Client(base_url=url, timeout=httpx.Timeout(timeout)) as client:
        health = client.get("/health")
        if health.status_code != 200:
            raise click.ClickException(
                f"{url}/health ответил {health.status_code}: {health.text[:200]}"
            )
        info = health.json()
        logger.info(
            f"сервис {url}: коллекция {info.get('collection')}, модель {info.get('model')}, второй уровень {info.get('resolver') or 'нет'}, каталог {info.get('catalog_rows')} строк"
        )
        rows: list[Row] = []
        with progress() as bar:
            task = bar.add_task("фото", total=len(pairs))
            for name, slug in pairs:
                rows.append(row_of(name, slug, *ask(client, images / name)))
                bar.update(task, advance=1)
    report = {"url": url, "health": info, "metrics": metrics(rows)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    with (out / "per_image.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(Row)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    print_report(report)
    logger.info(
        f"отчёт: {out}; accuracy {report['metrics']['accuracy']}, ошибок сервиса {report['metrics']['errors']}"
    )


if __name__ == "__main__":
    main()
