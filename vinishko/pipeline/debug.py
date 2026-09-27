"""Отладочный прогон одного фото или части датасета с выходами каждого этапа."""

import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import rich_click as click
from dotenv import load_dotenv
from kostyl.utils import setup_logger
from PIL import Image
from rich.console import Console
from rich.table import Table

from vinishko.pipeline.pipeline import Pipeline, PipelineResult
from vinishko.pipeline.steps.normalization.normalize import HERE as NORM_DIR
from vinishko.pipeline.steps.normalization.normalize import NormalizationReason, Normalizer, write_outputs
from vinishko.pipeline.steps.normalization.normalize import load_config as load_norm_config
from vinishko.pipeline.steps.normalization.seg import open_image
from vinishko.pipeline.steps.near_duplicates import NearDuplicateReranker
from vinishko.pipeline.steps.vanilla_vlm_rerank import VanillaVlmReranker
from vinishko.pipeline.steps.near_duplicates.rerank import NearDuplicateReason
from vinishko.pipeline.steps.vis_searcher import VisSearcher
from vinishko.pipeline.steps.vis_searcher import load_config as load_search_config
from vinishko.pipeline.steps.vis_searcher.configs import DEFAULT_CONFIG
from vinishko.pipeline.structs import BottleCandidates, BottleCrop, RejectedBottle, UnmatchedBottle

console = Console()
logger = setup_logger(fmt="detailed")
ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
InputMode = Literal["sam3", "cached", "raw"]


def normalization_markup(items: list[BottleCrop | RejectedBottle], stem: str, fmt: str) -> list[dict]:
    """Все бутылки нормализации для markup.json: годные с разметкой и именем файла кропа, отказы с причиной."""
    out = []
    for item in items:
        if isinstance(item, BottleCrop):
            out.append({"status": "ok", **item.markup(), "crop_file": f"{stem}_b{item.index}.{fmt}", "pipeline_crop_file": f"{stem}_b{item.index}_pipeline.npy", "crop_info": item.crop_info})
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


def output_path(output_dir: Path, image: Path) -> Path:
    """Безопасный путь результата по имени исходного изображения."""
    root = output_dir.resolve()
    out = (root / image.stem).resolve()
    if out == root or not out.is_relative_to(root) or image.resolve().is_relative_to(out):
        raise click.UsageError(f"недопустимая директория результата: {out}")
    return out


def prepare_output(output_dir: Path, image: Path, *, mode: InputMode = "sam3") -> Path:
    """Создаёт папку результата; cached/raw сохраняют прежнюю нормализацию, если она есть."""
    out = output_path(output_dir, image)
    if mode != "sam3":
        if mode == "cached" and not (out / "normalization" / "markup.json").is_file():
            raise click.UsageError(f"нет сохранённой нормализации для {image}: {out / 'normalization' / 'markup.json'}")
        previous = out / "result.json"
        if previous.is_file():
            try:
                old_image = json.loads(previous.read_text(encoding="utf-8"))["image"]
                if Path(old_image).resolve() != image.resolve():
                    raise ValueError(f"сохранённый результат относится к {old_image}")
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise click.UsageError(f"неверный {previous}: {exc}") from exc
        for name in ("search", "rerank"):
            target = out / name
            if target.exists():
                if not target.resolve().is_relative_to(out):
                    raise click.UsageError(f"недопустимая директория результата: {target}")
                shutil.rmtree(target)
        (out / "result.json").unlink(missing_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        return out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    return out


def cached_crop(row: dict, folder: Path, image: Path) -> BottleCrop:
    """Один кроп из lossless-кэша нового запуска или JPEG прежнего запуска."""
    crop_file = folder / row["crop_file"]
    if not crop_file.resolve().is_relative_to(folder.resolve()):
        raise ValueError(f"недопустимый путь кропа {crop_file}")
    meta = json.loads(crop_file.with_suffix(".json").read_text(encoding="utf-8"))
    if meta["uuid"] != row["uuid"] or Path(meta["source"]).resolve() != image.resolve():
        raise ValueError(f"кроп {crop_file} относится к другому фото или UUID")
    exact_name = row.get("pipeline_crop_file")
    if exact_name is not None:
        exact_file = folder / exact_name
        if not exact_file.resolve().is_relative_to(folder.resolve()):
            raise ValueError(f"недопустимый путь кропа {exact_file}")
        crop = np.load(exact_file, allow_pickle=False)
    else:
        with Image.open(crop_file) as saved:
            crop = np.asarray(saved.convert("RGB"))
    if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3:
        raise ValueError(f"кроп {crop_file} должен быть uint8 RGB")
    info = row.get("crop_info") or {"angle_deg": meta["bottle_angle"], "matrix_src_to_dst": meta["bottle_crop_matrix"]}
    return BottleCrop(row["index"], row["score"], row["bottle"], row["label"], row["angle"], crop, info, row["uuid"])


def load_normalization(image: Path, output_dir: Path) -> list[BottleCrop | RejectedBottle]:
    """Восстанавливает разметку и точные кропы из предыдущего запуска debug.py."""
    folder = output_path(output_dir, image) / "normalization"
    marker = folder / "markup.json"
    if not marker.is_file():
        raise click.UsageError(f"нет сохранённой нормализации для {image}: {marker}")
    try:
        rows = json.loads(marker.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("markup.json должен содержать список")
        items: list[BottleCrop | RejectedBottle] = []
        for row in rows:
            if row["status"] == "rejected":
                items.append(RejectedBottle(NormalizationReason(row["reason"]), row["detail"], row["score"], row["bottle"], row["label"], row["uuid"]))
            elif row["status"] == "ok":
                items.append(cached_crop(row, folder, image))
            else:
                raise ValueError(f"неизвестный статус {row['status']!r}")
        return items
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise click.ClickException(f"не удалось прочитать сохранённую нормализацию {marker}: {exc}") from exc


class CachedNormalizer:
    """Замена SAM3 при повторном поиске: читает результат debug.py по пути фото."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def __call__(self, image: Path) -> list[BottleCrop | RejectedBottle]:
        return load_normalization(image, self.output_dir)


class RawImageNormalizer:
    """Прямой поиск по целому фото: один синтетический кроп без сегментации и отбора."""

    def __call__(self, image: Image.Image) -> list[BottleCrop]:
        rgb = np.asarray(image.convert("RGB")).copy()
        width, height = image.size
        whole = [[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]]
        return [BottleCrop(1, 0.0, whole, [], 0.0, rgb, {"mode": "raw", "width": width, "height": height})]


def write_debug_result(
    image: Path,
    out: Path,
    img: Image.Image | Path,
    result: PipelineResult,
    norm_cfg: dict | None,
    searcher: VisSearcher | None,
    mode: InputMode,
) -> None:
    """Сохраняет ответ по фото и пишет нормализацию только после реального SAM3."""
    if mode == "cached":
        result.timings["normalization_cached"] = result.timings.pop("normalization")
    elif mode == "raw":
        result.timings["raw_input"] = result.timings.pop("normalization")
    else:
        if norm_cfg is None or not isinstance(img, Image.Image):
            raise RuntimeError("нет изображения или конфига для записи нормализации")
        norm_dir = out / "normalization"
        norm_dir.mkdir()
        write_outputs(image, norm_dir, np.asarray(img), result.normalization, norm_cfg)
        for item in result.normalization:
            if isinstance(item, BottleCrop):
                np.save(norm_dir / f"{image.stem}_b{item.index}_pipeline.npy", item.crop)
        (norm_dir / "markup.json").write_text(
            json.dumps(normalization_markup(result.normalization, image.stem, norm_cfg["output"]["format"]), ensure_ascii=False, indent=1), encoding="utf-8"
        )
    if searcher is not None and result.search:
        searcher.dump(result.search, out / "search")
    report = summary(image, result)
    report["input_mode"] = mode
    (out / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print_summary(report)
    logger.info(f"результаты: {out}")


def run_debug_batches(
    images: list[Path],
    output_dir: Path,
    pipeline: Pipeline,
    searcher: VisSearcher | None,
    reranker: NearDuplicateReranker | VanillaVlmReranker | None,
    norm_cfg: dict | None,
    mode: InputMode,
    batch_size: int,
) -> None:
    """Обрабатывает фото ограниченными пачками и записывает каждый готовый ответ."""
    for start in range(0, len(images), batch_size):
        batch_paths = images[start : start + batch_size]
        outputs = [prepare_output(output_dir, image, mode=mode) for image in batch_paths]
        batch_images = batch_paths if mode == "cached" else [open_image(image) for image in batch_paths]

        def before_rerank(index: int, dirs: list[Path] = outputs) -> None:
            if reranker is not None:
                reranker.trace_dir = dirs[index] / "rerank"

        def write_result(
            index: int,
            result: PipelineResult,
            paths: list[Path] = batch_paths,
            dirs: list[Path] = outputs,
            opened: list[Image.Image | Path] = batch_images,
        ) -> None:
            image, out, img = paths[index], dirs[index], opened[index]
            write_debug_result(image, out, img, result, norm_cfg, searcher, mode)

        pipeline.run_many(batch_images, before_rerank=before_rerank, on_result=write_result)


def input_mode(stage: str, skip_normalization: bool, no_normalization: bool, overrides: tuple[str, ...]) -> InputMode:
    """Два разных обхода SAM3: сохранённые кропы или исходное фото целиком."""
    if skip_normalization and no_normalization:
        raise click.UsageError("--skip-normalization и --no-normalization нельзя использовать вместе")
    if stage == "normalization" and (skip_normalization or no_normalization):
        raise click.UsageError("обход нормализации допустим только с --stage search или full")
    if (skip_normalization or no_normalization) and overrides:
        raise click.UsageError("--set действует только при запуске SAM3 без обхода нормализации")
    return "cached" if skip_normalization else "raw" if no_normalization else "sam3"


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output-dir", type=click.Path(file_okay=False, path_type=Path), required=True, help="Сюда ляжет директория с именем файла; старая с тем же именем удаляется")
@click.option("--stage", type=click.Choice(["normalization", "search", "full"]), default="full", show_default=True, help="Последний выполняемый этап: нормализация, векторный поиск или NDR")
@click.option("--limit", type=click.IntRange(min=1), default=None, help="Первые N фото датасета или директории; без лимита — все")
@click.option("--batch-size", type=click.IntRange(min=1), default=2, show_default=True, help="Число фото в пачке поиска; нормализация SAM3 всегда по одному фото")
@click.option("--skip-normalization", is_flag=True, help="Для search/full использовать <output-dir>/<имя>/normalization из прежнего запуска, не загружая SAM3")
@click.option("--no-normalization", is_flag=True, help="Для search/full искать прямо по каждому исходному фото целиком, без SAM3 и сохранённых кропов")
@click.option("--no-search", is_flag=True, help="Синоним --stage normalization")
@click.option("--search-config", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=DEFAULT_CONFIG, show_default=True, help="Конфиг визуального поиска; его debug_path не используется, разбор пишется в <output-dir>/<имя>/search")
@click.option("-c", "--norm-config", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=NORM_DIR / "normalize.toml", show_default=True, help="Конфиг нормализации")
@click.option("--set", "overrides", multiple=True, metavar="секция.ключ=значение", help="Переопределить параметр конфига нормализации")
def main(input_path: Path, output_dir: Path, stage: str, limit: int | None, batch_size: int, skip_normalization: bool, no_normalization: bool, no_search: bool, search_config: Path, norm_config: Path, overrides: tuple[str, ...]) -> None:
    """Прогнать выбранные фото до указанного этапа и записать выходы по директориям.

    <output-dir>/<имя файла>/normalization — кропы годных бутылок, маски и json как у CLI нормализации, плюс markup.json со всеми
    бутылками и причинами отказов; search — разбор поиска в формате debug_path: кроп запроса, картинки кандидатов с косинусом
    и группой в имени, results.json; result.json — итог по каждой бутылке и время шагов. Устройства — NORMALIZER_DEV и VIS_SEARCHER_DEV.
    """
    load_dotenv(ROOT / ".env", override=False)
    if no_search:
        stage = "normalization"
    mode = input_mode(stage, skip_normalization, no_normalization, overrides)
    images = select_images(input_path, limit)
    if mode == "cached":
        for image in images:
            marker = output_path(output_dir, image) / "normalization" / "markup.json"
            if not marker.is_file():
                raise click.UsageError(f"нет сохранённой нормализации для {image}: {marker}")
    norm_cfg = load_norm_config(norm_config, list(overrides)) if mode == "sam3" else None
    normalizer = Normalizer(norm_cfg, norm_config.resolve().parent) if mode == "sam3" else CachedNormalizer(output_dir) if mode == "cached" else RawImageNormalizer()
    searcher = None
    reranker = None
    if stage != "normalization":
        cfg = load_search_config(search_config)
        cfg.debug_path = None
        try:
            searcher = VisSearcher(cfg)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    pipeline = Pipeline(normalizer, searcher, enable_rerank=stage == "full")
    reranker = pipeline.reranker
    search_batch_size = batch_size if searcher is not None else 1
    logger.info(f"этап {stage}; изображений {len(images)}; фото в пачке {search_batch_size}; вход {mode}")
    run_debug_batches(images, output_dir, pipeline, searcher, reranker, norm_cfg, mode, search_batch_size)


if __name__ == "__main__":
    main()
