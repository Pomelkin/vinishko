"""Оценка пайплайна с выходами каждого шага на диске: нормализация → поиск → выбор позиции.

Тестовый набор: python -m vinishko.pipeline.evaluate_pipeline --test-dir datasets/hack-vine/test — метрики поиска как у
vis_searcher.evaluate плюс классификация итогового ответа, отчёт в reports/pipeline/<время>. Одно фото без разметки:
python -m vinishko.pipeline.evaluate_pipeline --image photo.jpg -o runs — только выходы шагов и итог в терминал.
Разметка — CSV с колонками image_filename и slug; пустой slug значит, что верный ответ — отказ. Сквозная проверка через API,
без промежуточных картинок, — vinishko/e2e.py.
"""

import csv
import json
import shutil
import statistics
from collections import Counter
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from datetime import UTC
from datetime import datetime
from pathlib import Path

import numpy as np
import rich_click as click
from kostyl.utils import setup_logger
from rich.console import Console
from rich.table import Table

from vinishko.pipeline.pipeline import Pipeline
from vinishko.pipeline.pipeline import PipelineResult
from vinishko.pipeline.steps.near_duplicates import NearDuplicateResolver
from vinishko.pipeline.steps.normalization.normalize import HERE as NORM_DIR
from vinishko.pipeline.steps.normalization.normalize import Normalizer
from vinishko.pipeline.steps.normalization.normalize import (
    load_config as load_norm_config,
)
from vinishko.pipeline.steps.normalization.normalize import write_outputs
from vinishko.pipeline.steps.normalization.seg import open_image
from vinishko.pipeline.steps.vis_searcher.catalog import FIELD_GROUP_SLUGS
from vinishko.pipeline.steps.vis_searcher.configs import DEFAULT_CONFIG
from vinishko.pipeline.steps.vis_searcher.configs import load_config
from vinishko.pipeline.steps.vis_searcher.evaluate import DUMPS
from vinishko.pipeline.steps.vis_searcher.evaluate import KS
from vinishko.pipeline.steps.vis_searcher.evaluate import Row as SearchRow
from vinishko.pipeline.steps.vis_searcher.evaluate import catalog_slugs
from vinishko.pipeline.steps.vis_searcher.evaluate import evaluate_image
from vinishko.pipeline.steps.vis_searcher.evaluate import is_miss
from vinishko.pipeline.steps.vis_searcher.evaluate import metrics
from vinishko.pipeline.steps.vis_searcher.evaluate import print_metrics
from vinishko.pipeline.steps.vis_searcher.evaluate import progress
from vinishko.pipeline.steps.vis_searcher.evaluate import read_test
from vinishko.pipeline.steps.vis_searcher.evaluate import share
from vinishko.pipeline.steps.vis_searcher.search import VisSearcher
from vinishko.pipeline.structs import BottleCandidates
from vinishko.pipeline.structs import BottleCrop
from vinishko.pipeline.structs import MatchedBottle
from vinishko.pipeline.structs import RejectedBottle


FINAL = ("matched", "rejected_search", "rejected_resolve", "no_bottle")
"""Итог по фото: выбрана позиция; отказ поиска; отказ второго уровня; нормализация не нашла годной бутылки."""
SOURCE_SEARCH_TOP1 = "search_top1"
"""answer_source без второго уровня: ответом считается top-1 поиска."""
console = Console()
logger = setup_logger(fmt="detailed")


@dataclass(slots=True)
class Row(SearchRow):
    """Итог по фото: поля поиска из vis_searcher.evaluate плюс итоговый ответ."""

    answer: str
    """Итоговый slug; пусто — отказ."""
    answer_source: str
    """group — модель выбрала внутри группы, других финалистов не было; final — выбрала в финале; search_top1 — без второго уровня; пусто — отказ."""
    model_calls: int
    """Вызовов модели второго уровня на фото, по всем бутылкам: файлов trace в resolve/."""
    final: str
    """Одно из FINAL."""
    correct: bool | None
    """Ответ верный: slug совпал, либо отказ на фото без ответа; None для фото, чей ответ не в коллекции."""
    group_correct: bool | None
    """Выбранная позиция из группы ожидаемого slug; None, если ответа нет или позиция не выбрана."""


def decide(result: PipelineResult) -> tuple[str, str, str, list[str]]:
    """Итог по бутылке с лучшим скором отбора: (final, slug, источник, slug группы выбранной позиции)."""
    if not result.search:
        return "no_bottle", "", "", []
    verdict = result.resolution[0] if result.resolution else result.search[0]
    if isinstance(verdict, MatchedBottle):
        return (
            "matched",
            verdict.candidate.slug,
            verdict.source,
            list(verdict.candidate.payload.get(FIELD_GROUP_SLUGS, [])),
        )
    if isinstance(verdict, BottleCandidates):
        top = verdict.candidates[0]
        return (
            "matched",
            top.slug,
            SOURCE_SEARCH_TOP1,
            list(top.payload.get(FIELD_GROUP_SLUGS, [])),
        )
    return f"rejected_{verdict.rejection.stage}", "", "", []


def e2e_row(search_row: SearchRow, result: PipelineResult, model_calls: int) -> Row:
    """Строка отчёта: поля поиска плюс итог."""
    final, answer, source, group = decide(result)
    if search_row.in_catalog:
        correct: bool | None = answer == search_row.slug
        group_correct: bool | None = (
            search_row.slug in group if final == "matched" else None
        )
    elif not search_row.slug:
        correct, group_correct = final != "matched", None
    else:
        correct, group_correct = None, None
    return Row(
        **asdict(search_row),
        answer=answer,
        answer_source=source,
        model_calls=model_calls,
        final=final,
        correct=correct,
        group_correct=group_correct,
    )


def e2e_metrics(rows: list[Row], timings: list[dict[str, float]]) -> dict:
    """Метрики классификации итогового ответа: система выдаёт один slug либо отказ, и это сравнивается с разметкой.

    accuracy — верный slug либо верный отказ на фото без ответа, по всем фото, кроме тех, чей ответ не в коллекции (они не могут быть
    верными по построению); accuracy_all_images — то же, но такие фото считаются ошибкой, это честный счёт на тесте. precision — доля
    верных среди выданных ответов, recall — доля верных среди фото с ответом, f1 — их гармоническое среднее; confusion — исходы в штуках.
    По фото с ответом: group_of_answer, wrong_slug, ложные отказы по шагам. По фото без ответа: correct_reject по шагам и false_accept.
    model_calls — вызовов модели второго уровня всего и на фото, answers_by_source — кто выбрал выданные ответы.
    """
    with_answer = [r for r in rows if r.in_catalog]
    without = [r for r in rows if not r.slug]
    unindexed = [r for r in rows if r.slug and not r.in_catalog]
    answered = [r for r in rows if r.final == "matched"]
    correct_answers = sum(bool(r.correct) for r in with_answer)
    correct_rejects = sum(r.final != "matched" for r in without)
    precision = correct_answers / len(answered) if answered else None
    recall = correct_answers / len(with_answer) if with_answer else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    stages = {
        name: round(statistics.mean(t[name] for t in timings if name in t), 3)
        for name in ("normalization", "search", "resolve")
        if any(name in t for t in timings)
    }
    return {
        "images": len(rows),
        "accuracy": share(bool(r.correct) for r in with_answer + without),
        "accuracy_all_images": round((correct_answers + correct_rejects) / len(rows), 4)
        if rows
        else None,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "confusion": {
            "answer_correct": correct_answers,
            "answer_wrong": sum(
                r.final == "matched" and not r.correct for r in with_answer
            ),
            "answer_on_empty": sum(r.final == "matched" for r in without),
            "answer_not_indexed": sum(r.final == "matched" for r in unindexed),
            "reject_with_answer": sum(r.final != "matched" for r in with_answer),
            "reject_on_empty": correct_rejects,
            "reject_not_indexed": sum(r.final != "matched" for r in unindexed),
        },
        "group_of_answer": share(bool(r.group_correct) for r in with_answer),
        "wrong_slug": share(
            r.final == "matched" and not r.correct for r in with_answer
        ),
        "false_reject_search": share(r.final == "rejected_search" for r in with_answer),
        "false_reject_resolve": share(
            r.final == "rejected_resolve" for r in with_answer
        ),
        "no_bottle": share(r.final == "no_bottle" for r in with_answer),
        "correct_reject": share(r.final != "matched" for r in without),
        "correct_reject_search": share(r.final == "rejected_search" for r in without),
        "correct_reject_resolve": share(r.final == "rejected_resolve" for r in without),
        "false_accept": share(r.final == "matched" for r in without),
        "model_calls": sum(r.model_calls for r in rows),
        "model_calls_per_image": round(statistics.mean(r.model_calls for r in rows), 3)
        if rows
        else None,
        "answers_by_source": dict(Counter(r.answer_source for r in answered)),
        "seconds_per_image": round(statistics.mean(r.seconds for r in rows), 3)
        if rows
        else None,
        "seconds_per_stage": stages,
    }


def normalization_markup(
    items: list[BottleCrop | RejectedBottle], stem: str, fmt: str
) -> list[dict]:
    """Все бутылки нормализации для markup.json: годные с разметкой и именами файлов кропов, отказы с причиной."""
    out = []
    for item in items:
        if isinstance(item, BottleCrop):
            out.append(
                {
                    "status": "ok",
                    **item.markup(),
                    "crop_file": f"{stem}_b{item.index}.{fmt}",
                    "box_file": f"{stem}_b{item.index}_box.{fmt}",
                }
            )
        else:
            out.append(
                {
                    "status": "rejected",
                    **asdict(item),
                    "stage": item.rejection.stage,
                    "label": item.rejection.label,
                    "message": item.message,
                }
            )
    return out


def summary(image: Path, result: PipelineResult) -> dict:
    """result.json: итог по каждой бутылке после всех шагов и время шагов."""
    found = {
        r.crop.uuid: len(r.candidates)
        for r in result.search
        if isinstance(r, BottleCandidates)
    }
    bottles = []
    for bottle in result.bottles:
        entry: dict = {
            "uuid": bottle.uuid,
            "status": bottle.status,
            "bottle_score": bottle.score,
        }
        if bottle.rejection is not None:
            entry |= {
                "stage": bottle.rejection.stage,
                "reason": bottle.rejection.reason,
                "message": bottle.rejection.message,
            }
        elif bottle.match is not None:
            answer = bottle.match
            entry |= {
                "slug": answer.candidate.slug,
                "group": answer.candidate.group,
                "score": answer.candidate.score,
                "source": answer.source,
                "checklist": answer.checklist,
                "candidates": found.get(bottle.uuid, 0),
            }
        elif bottle.candidates:
            entry["candidates"] = [
                {
                    "slug": c.slug,
                    "group": c.group,
                    "score": c.score,
                    "scores": c.scores,
                    "retrieved": c.retrieved,
                }
                for c in bottle.candidates
            ]
        bottles.append(entry)
    return {
        "image": str(image),
        "timings_s": {k: round(v, 3) for k, v in result.timings.items()},
        "bottles": bottles,
    }


def print_summary(report: dict) -> None:
    """Итог по одному фото в терминал: строка на бутылку."""
    table = Table(
        title=f"{Path(report['image']).name} · "
        + ", ".join(f"{k} {v:.2f} с" for k, v in report["timings_s"].items())
    )
    table.add_column("бутылка")
    table.add_column("итог")
    table.add_column("подробности")
    for bottle in report["bottles"]:
        if bottle["status"] == "rejected":
            table.add_row(
                bottle["uuid"][:8], f"отказ ({bottle['stage']})", bottle["message"]
            )
        elif bottle["status"] == "matched":
            table.add_row(
                bottle["uuid"][:8],
                f"выбрана 1 из {bottle['candidates']}",
                f"{bottle['slug']} · cos {bottle['score']:.3f} · группа {bottle['group']} · {bottle['source']}",
            )
        elif bottle["status"] == "candidates":
            top = bottle["candidates"][0]
            table.add_row(
                bottle["uuid"][:8],
                f"кандидатов {len(bottle['candidates'])}",
                f"top-1 {top['slug']} · cos {top['score']:.3f} · группа {top['group']}",
            )
        else:
            table.add_row(
                bottle["uuid"][:8],
                "годная, поиск не запускался",
                f"скор отбора {bottle['bottle_score']}",
            )
    console.print(table)


def print_e2e(report: dict) -> None:
    """Сводка итогового ответа в терминал."""
    table = Table(
        title=f"итог пайплайна · второй уровень: {report['resolver'] or 'нет, ответ — top-1 поиска'}"
    )
    table.add_column("метрика")
    table.add_column("значение", justify="right")
    for name, value in report["metrics"]["e2e"].items():
        table.add_row(name, format_metric(name, value))
    console.print(table)


def format_metric(name: str, value: object) -> str:
    """Значение метрики строкой: доли в процентах, словари через запятую."""
    if value is None:
        return "—"
    if isinstance(value, dict):
        return ", ".join(f"{k} {v}" for k, v in value.items())
    if isinstance(value, float) and name != "seconds_per_image":
        return f"{value:.1%}"
    return str(value)


def dump_image(
    out: Path,
    path: Path,
    result: PipelineResult,
    norm_cfg: dict,
    searcher: VisSearcher | None,
) -> None:
    """Папка одного фото: исходник, normalization/ как у CLI нормализации плюс markup.json, search/ как при debug_path, result.json; resolve/ пишет сам второй уровень."""
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, out / f"source{path.suffix.lower()}")
    norm_dir = out / "normalization"
    norm_dir.mkdir(exist_ok=True)
    write_outputs(
        path, norm_dir, np.asarray(open_image(path)), result.normalization, norm_cfg
    )
    (norm_dir / "markup.json").write_text(
        json.dumps(
            normalization_markup(
                result.normalization, path.stem, norm_cfg["output"]["format"]
            ),
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    if result.search and searcher is not None:
        searcher.dump(result.search, out / "search")
    (out / "result.json").write_text(
        json.dumps(summary(path, result), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def is_e2e_miss(row: Row) -> bool:
    """Фото, на которое стоит посмотреть глазами: промах поиска либо неверный итог."""
    return is_miss(row) or row.correct is False


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
    "--image",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Одно фото без разметки: выходы шагов в <out>/<имя>, метрик нет",
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
@click.option(
    "--no-search",
    is_flag=True,
    help="Только нормализация: без qdrant, модели поиска и второго уровня",
)
@click.option(
    "--no-resolve",
    is_flag=True,
    help="Без второго уровня: итоговый ответ — top-1 поиска, модель не вызывается",
)
@click.option(
    "--limit", type=click.IntRange(min=1), default=None, help="Первые N фото разметки"
)
@click.option(
    "--dump",
    type=click.Choice(DUMPS),
    default="all",
    show_default=True,
    help="Папка на фото в <out>/dumps/<фото>: все фото, только промахи либо ничего",
)
@click.option(
    "-o",
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Куда писать report.json, per_image.csv и дампы; по умолчанию reports/pipeline/<время>, для --image — runs/",
)
def main(
    test_dir: Path,
    csv_path: Path | None,
    images: Path | None,
    image: Path | None,
    config_path: Path,
    norm_config: Path,
    overrides: tuple[str, ...],
    no_search: bool,
    no_resolve: bool,
    limit: int | None,
    dump: str,
    out: Path | None,
) -> None:
    """Прогнать фото через весь пайплайн и разложить выходы шагов по папкам.

    На фото: normalization/ — кропы годных бутылок, маски и json как у CLI нормализации плюс markup.json со всеми бутылками и причинами
    отказов; search/ — разбор поиска: кропы запроса, картинки кандидатов со скором и группой в имени, results.json; resolve/<uuid бутылки>/ —
    ответ модели второго уровня на каждый вызов: group<N>.json по группам, final.json; result.json — итог по каждой бутылке и время шагов. По тестовому набору сверх того report.json
    с метриками поиска и итогового ответа и per_image.csv. Устройства — NORMALIZER_DEV и VIS_SEARCHER_DEV, ключ модели — OPENROUTER_API_KEY.
    """
    if image is not None:
        pairs, images, out = [(image.name, "")], image.parent, out or Path("runs")
    else:
        pairs = read_test(csv_path or test_dir / "test.csv")[:limit]
        images = images or test_dir / "images"
        missing = [name for name, _ in pairs if not (images / name).is_file()]
        if missing:
            raise click.UsageError(
                f"в {images} нет {len(missing)} фото из разметки, например {missing[:3]}"
            )
        out = out or Path("reports/pipeline") / datetime.now(
            tz=UTC
        ).astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    out.mkdir(parents=True, exist_ok=True)
    norm_cfg = load_norm_config(norm_config, list(overrides))
    cfg = load_config(config_path)
    cfg.debug_path = None
    searcher = VisSearcher(cfg) if not no_search else None
    resolver = (
        NearDuplicateResolver() if searcher is not None and not no_resolve else None
    )
    pipeline = Pipeline(
        Normalizer(norm_cfg, norm_config.resolve().parent),
        searcher,
        resolver,
        search=not no_search,
        resolve=not no_resolve,
    )
    known = catalog_slugs(searcher) if searcher is not None and image is None else set()
    if image is None:
        logger.info(
            f"тест: {len(pairs)} фото, с ответом в коллекции {sum(slug in known for _, slug in pairs)}, без ответа {sum(not slug for _, slug in pairs)}"
            + (
                f"; коллекция {searcher.info.count} позиций"
                if searcher is not None
                else "; без поиска"
            )
            + (
                f"; второй уровень: {resolver_name(resolver)}"
                if resolver is not None
                else "; без второго уровня"
            )
        )
    rows, timings = run(
        pipeline,
        pairs,
        images,
        out,
        known,
        norm_cfg,
        dump if image is None else "all",
        single=image is not None,
    )
    if image is not None:
        logger.info(f"результаты: {out / image.stem}")
        return
    report = {
        "collection": searcher.info.name if searcher is not None else None,
        "model": f"{searcher.files.repo}@{searcher.files.revision[:8]}"
        if searcher is not None
        else None,
        "encoder_input": list(cfg.encoder_input),
        "mode": cfg.search.mode,
        "search": cfg.search.model_dump(),
        "top_k": cfg.top_k,
        "device": str(searcher.device) if searcher is not None else None,
        "resolver": resolver_name(resolver) if resolver is not None else None,
        "metrics": {"search": metrics(rows), "e2e": e2e_metrics(rows, timings)},
    }
    write_report(out, report, rows)
    if searcher is not None:
        print_metrics(
            {
                "collection": report["collection"],
                "mode": report["mode"],
                "top_k": report["top_k"],
                "metrics": report["metrics"]["search"],
            }
        )
    print_e2e(report)
    logger.info(
        f"отчёт: {out}; recall@{KS[0]} поиска {report['metrics']['search']['recall@1']}, accuracy итога {report['metrics']['e2e']['accuracy']}"
    )


def resolver_name(resolver: NearDuplicateResolver) -> str:
    """Модель второго уровня для отчёта."""
    return resolver.settings.openrouter.model or "модель из OPENROUTER_MODEL"


def run(
    pipeline: Pipeline,
    pairs: list[tuple[str, str]],
    images: Path,
    out: Path,
    known: set[str],
    norm_cfg: dict,
    dump: str,
    single: bool,
) -> tuple[list[Row], list[dict[str, float]]]:
    """Прогон фото по одному: строка отчёта, время шагов и, по режиму dump, папка с выходами шагов; trace второго уровня уходит в неё же."""
    searcher = pipeline.searcher if isinstance(pipeline.searcher, VisSearcher) else None
    resolver = (
        pipeline.resolver
        if isinstance(pipeline.resolver, NearDuplicateResolver)
        else None
    )
    rows: list[Row] = []
    timings: list[dict[str, float]] = []
    with progress() as bar:
        task = bar.add_task("фото", total=len(pairs))
        for name, slug in pairs:
            path = images / name
            dump_dir = out / path.stem if single else out / "dumps" / path.stem
            if dump_dir.exists():
                shutil.rmtree(dump_dir)
            if resolver is not None:
                pipeline.resolver = resolver.with_trace_dir(dump_dir / "resolve")
            search_row, result = evaluate_image(pipeline, path, slug, known)
            calls = len(list((dump_dir / "resolve").glob("*/*.json")))
            row = e2e_row(search_row, result, calls)
            rows.append(row)
            timings.append(result.timings)
            if dump == "all" or (dump == "misses" and is_e2e_miss(row)):
                dump_image(dump_dir, path, result, norm_cfg, searcher)
            elif dump_dir.exists():
                shutil.rmtree(dump_dir)
            if single:
                print_summary(summary(path, result))
            bar.update(task, advance=1)
    return rows, timings


def write_report(out: Path, report: dict, rows: list[Row]) -> None:
    """report.json и per_image.csv."""
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    with (out / "per_image.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(Row)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


if __name__ == "__main__":
    main()
