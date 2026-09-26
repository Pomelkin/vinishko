"""Отладочный прогон одного фото или части датасета с выходами каждого этапа."""

import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import rich_click as click
from dotenv import load_dotenv
from kostyl.utils import setup_logger
from rich.console import Console
from rich.table import Table

from vinishko.pipeline.pipeline import Pipeline, PipelineResult
from vinishko.pipeline.steps.normalization.normalize import HERE as NORM_DIR
from vinishko.pipeline.steps.normalization.normalize import Normalizer, write_outputs
from vinishko.pipeline.steps.normalization.normalize import load_config as load_norm_config
from vinishko.pipeline.steps.normalization.seg import open_image
from vinishko.pipeline.steps.near_duplicates import NearDuplicateReranker
from vinishko.pipeline.steps.near_duplicates.rerank import NearDuplicateReason
from vinishko.pipeline.steps.vis_searcher import VisSearcher
from vinishko.pipeline.steps.vis_searcher import load_config as load_search_config
from vinishko.pipeline.steps.vis_searcher.configs import DEFAULT_CONFIG
from vinishko.pipeline.structs import BottleCandidates, BottleCrop, RejectedBottle, UnmatchedBottle

console = Console()
logger = setup_logger(fmt="detailed")
ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def normalization_markup(items: list[BottleCrop | RejectedBottle], stem: str, fmt: str) -> list[dict]:
    """Все бутылки нормализации для markup.json: годные с разметкой и именем файла кропа, отказы с причиной."""
    out = []
    for item in items:
        if isinstance(item, BottleCrop):
            out.append({"status": "ok", **item.markup(), "crop_file": f"{stem}_b{item.index}.{fmt}"})
        else:
            out.append({"status": "rejected", **asdict(item), "title": item.reason.title, "message": item.message})
    return out


def summary(image: Path, result: PipelineResult) -> dict:
    """result.json: итог по каждой бутылке после всех шагов и время шагов."""
    search_rejections = {r.rejected.uuid: r for r in result.search if isinstance(r, UnmatchedBottle)}
    found = {r.crop.uuid: r for r in result.search if isinstance(r, BottleCandidates)}
    bottles = []
    for item in result.items:
        if isinstance(item, RejectedBottle):
            step = "rerank" if item.reason == NearDuplicateReason.NOT_FOUND else ("search" if item.uuid in search_rejections else "normalization")
            bottles.append({"uuid": item.uuid, "status": "rejected", "step": step, "reason": item.reason, "message": item.message, "score": item.score, "selection": search_rejections[item.uuid].selection if item.uuid in search_rejections else None})
        elif item.uuid in found:
            candidates = found[item.uuid].candidates
            bottles.append({
                "uuid": item.uuid,
                "status": "candidates",
                "bottle_score": item.score,
                "candidates": [{"slug": c.slug, "group": c.group, "score": c.score, "retrieved": c.retrieved} for c in candidates],
                "selection": found[item.uuid].selection,
            })
        else:
            bottles.append({"uuid": item.uuid, "status": "normalized", "bottle_score": item.score})
    return {"image": str(image), "timings_s": {k: round(v, 3) for k, v in result.timings.items()}, "bottles": bottles}


def print_summary(report: dict) -> None:
    """Итог в терминал: строка на бутылку."""
    table = Table(title=f"{Path(report['image']).name} · " + ", ".join(f"{k} {v:.2f} с" for k, v in report["timings_s"].items()))
    table.add_column("бутылка")
    table.add_column("итог")
    table.add_column("подробности")
    for bottle in report["bottles"]:
        if bottle["status"] == "rejected":
            table.add_row(bottle["uuid"][:8], f"отказ ({bottle['step']})", bottle["message"])
        elif bottle["status"] == "candidates":
            top = bottle["candidates"][0]
            table.add_row(bottle["uuid"][:8], f"кандидатов {len(bottle['candidates'])}", f"top-1 {top['slug']} · cos {top['score']:.3f} · группа {top['group']}")
        else:
            table.add_row(bottle["uuid"][:8], "годная, поиск не запускался", f"скор отбора {bottle['bottle_score']}")
    console.print(table)


def select_images(input_path: Path, limit: int | None) -> list[Path]:
    """Фото из файла, датасета с CSV или директории images; порядок CSV либо имени файла."""
    if input_path.is_file():
        images = [input_path]
    else:
        manifest = next((input_path / name for name in ("test.csv", "catalog.csv") if (input_path / name).is_file()), None)
        if manifest is not None:
            with manifest.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if "image_filename" not in (reader.fieldnames or []):
                    raise click.UsageError(f"в {manifest} нет колонки image_filename")
                images = [input_path / "images" / name for row in reader if (name := (row["image_filename"] or "").strip())]
        else:
            folder = input_path / "images" if (input_path / "images").is_dir() else input_path
            images = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=lambda p: p.name.casefold())
    selected = images[:limit]
    if not selected:
        raise click.UsageError(f"в {input_path} нет изображений")
    missing = [str(path) for path in selected if not path.is_file()]
    if missing:
        raise click.UsageError(f"не найдены изображения из CSV: {missing[:3]}")
    return selected


def prepare_output(output_dir: Path, image: Path) -> Path:
    """Создаёт папку результата после проверки границ рекурсивного удаления."""
    root = output_dir.resolve()
    out = (root / image.stem).resolve()
    if out == root or not out.is_relative_to(root) or image.resolve().is_relative_to(out):
        raise click.UsageError(f"недопустимая директория результата: {out}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    return out


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output-dir", type=click.Path(file_okay=False, path_type=Path), required=True, help="Сюда ляжет директория с именем файла; старая с тем же именем удаляется")
@click.option("--stage", type=click.Choice(["normalization", "search", "full"]), default="full", show_default=True, help="Последний выполняемый этап: нормализация, векторный поиск или NDR")
@click.option("--limit", type=click.IntRange(min=1), default=None, help="Первые N фото датасета или директории; без лимита — все")
@click.option("--no-search", is_flag=True, help="Синоним --stage normalization")
@click.option("--search-config", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=DEFAULT_CONFIG, show_default=True, help="Конфиг визуального поиска; его debug_path не используется, разбор пишется в <output-dir>/<имя>/search")
@click.option("-c", "--norm-config", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=NORM_DIR / "normalize.toml", show_default=True, help="Конфиг нормализации")
@click.option("--set", "overrides", multiple=True, metavar="секция.ключ=значение", help="Переопределить параметр конфига нормализации")
def main(input_path: Path, output_dir: Path, stage: str, limit: int | None, no_search: bool, search_config: Path, norm_config: Path, overrides: tuple[str, ...]) -> None:
    """Прогнать выбранные фото до указанного этапа и записать выходы по директориям.

    <output-dir>/<имя файла>/normalization — кропы годных бутылок, маски и json как у CLI нормализации, плюс markup.json со всеми
    бутылками и причинами отказов; search — разбор поиска в формате debug_path: кроп запроса, картинки кандидатов с косинусом
    и группой в имени, results.json; result.json — итог по каждой бутылке и время шагов. Устройства — NORMALIZER_DEV и VIS_SEARCHER_DEV.
    """
    load_dotenv(ROOT / ".env", override=False)
    if no_search:
        stage = "normalization"
    images = select_images(input_path, limit)
    norm_cfg = load_norm_config(norm_config, list(overrides))
    normalizer = Normalizer(norm_cfg, norm_config.resolve().parent)
    searcher = None
    reranker = None
    if stage != "normalization":
        cfg = load_search_config(search_config)
        cfg.debug_path = None
        searcher = VisSearcher(cfg)
        if stage == "full":
            reranker = NearDuplicateReranker.from_search_config(cfg)
    pipeline = Pipeline(normalizer, searcher, reranker, enable_rerank=stage == "full")
    logger.info(f"этап {stage}; изображений {len(images)}")
    for image in images:
        out = prepare_output(output_dir, image)
        if reranker is not None:
            reranker.trace_dir = out / "rerank"
        img = open_image(image)
        result = pipeline(img)

        norm_dir = out / "normalization"
        norm_dir.mkdir()
        write_outputs(image, norm_dir, np.asarray(img), result.normalization, norm_cfg)
        (norm_dir / "markup.json").write_text(
            json.dumps(normalization_markup(result.normalization, image.stem, norm_cfg["output"]["format"]), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        if searcher is not None and result.search:
            searcher.dump(result.search, out / "search")
        report = summary(image, result)
        (out / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print_summary(report)
        logger.info(f"результаты: {out}")


if __name__ == "__main__":
    main()
