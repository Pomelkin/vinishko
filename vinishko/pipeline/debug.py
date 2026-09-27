"""Отладочный прогон одного фото или части датасета с выходами каждого этапа."""

import csv
import json
import shutil
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
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
from vinishko.pipeline.steps.near_duplicates.rerank import NDR_CONCURRENCY, NearDuplicateReason
from vinishko.pipeline.steps.vis_searcher import VisSearcher
from vinishko.pipeline.steps.vis_searcher import load_config as load_search_config
from vinishko.pipeline.steps.vis_searcher.catalog import FIELD_GROUP_SLUGS
from vinishko.pipeline.steps.vis_searcher.configs import DEFAULT_CONFIG
from vinishko.pipeline.steps.vis_searcher.search import SearchReason
from vinishko.pipeline.structs import BottleCandidates, BottleCrop, Candidate, RejectedBottle, UnmatchedBottle

console = Console()
logger = setup_logger(fmt="detailed")
ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
InputMode = Literal["sam3", "cached", "raw", "precomputed"]


def normalization_markup(items: list[BottleCrop | RejectedBottle], stem: str, fmt: str) -> list[dict]:
    """Все бутылки нормализации для markup.json: годные с разметкой и именем файла кропа, отказы с причиной."""
    out = []
    for item in items:
        if isinstance(item, BottleCrop):
            out.append({"status": "ok", **item.markup(), "crop_file": f"{stem}_b{item.index}.{fmt}", "pipeline_crop_file": f"{stem}_b{item.index}_pipeline.npy", "crop_info": item.crop_info, "box_file": f"{stem}_b{item.index}_box.{fmt}", "pipeline_box_file": f"{stem}_b{item.index}_box_pipeline.npy", "box_info": item.box_info})
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


def cached_crop(row: dict, folder: Path, image: Path, original: np.ndarray | None = None) -> BottleCrop:
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
    box_name = row.get("pipeline_box_file") or row.get("box_file")
    if box_name:
        box_file = folder / box_name
        if not box_file.resolve().is_relative_to(folder.resolve()):
            raise ValueError(f"недопустимый путь кропа {box_file}")
        if box_file.suffix == ".npy":
            box = np.load(box_file, allow_pickle=False)
        else:
            with Image.open(box_file) as saved:
                box = np.asarray(saved.convert("RGB"))
    else:
        box = crop  # старый кэш содержал только кроп для поиска
    if box.dtype != np.uint8 or box.ndim != 3 or box.shape[2] != 3:
        raise ValueError(f"кроп {box_name} должен быть uint8 RGB")
    return BottleCrop(row["index"], row["score"], row["bottle"], row["label"], row["angle"], crop, info, box, row.get("box_info") or {}, row["uuid"], original)


def load_normalization(image: Path, output_dir: Path, *, include_original: bool = False) -> list[BottleCrop | RejectedBottle]:
    """Восстанавливает разметку и точные кропы из предыдущего запуска debug.py."""
    folder = output_path(output_dir, image) / "normalization"
    marker = folder / "markup.json"
    if not marker.is_file():
        raise click.UsageError(f"нет сохранённой нормализации для {image}: {marker}")
    try:
        rows = json.loads(marker.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("markup.json должен содержать список")
        original = None
        if include_original:
            original_image = open_image(image)
            original_image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            original = np.asarray(original_image).copy()
        items: list[BottleCrop | RejectedBottle] = []
        for row in rows:
            if row["status"] == "rejected":
                items.append(RejectedBottle(NormalizationReason(row["reason"]), row["detail"], row["score"], row["bottle"], row["label"], row["uuid"]))
            elif row["status"] == "ok":
                items.append(cached_crop(row, folder, image, original))
            else:
                raise ValueError(f"неизвестный статус {row['status']!r}")
        return items
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise click.ClickException(f"не удалось прочитать сохранённую нормализацию {marker}: {exc}") from exc


class CachedNormalizer:
    """Замена SAM3 при повторном поиске: читает результат debug.py по пути фото."""

    def __init__(self, output_dir: Path, *, include_original: bool = False) -> None:
        self.output_dir = output_dir
        self.include_original = include_original

    def __call__(self, image: Path) -> list[BottleCrop | RejectedBottle]:
        return load_normalization(image, self.output_dir, include_original=self.include_original)


class RawImageNormalizer:
    """Прямой поиск по целому фото: один синтетический кроп без сегментации и отбора."""

    def __call__(self, image: Image.Image) -> list[BottleCrop]:
        rgb = np.asarray(image.convert("RGB")).copy()
        width, height = image.size
        whole = [[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]]
        return [BottleCrop(1, 0.0, whole, [], 0.0, rgb, {"mode": "raw", "width": width, "height": height}, rgb, {"mode": "raw"}, original=rgb)]


def precomputed_file(image: Path, folder: Path) -> Path | None:
    """S3-кропы теста названы по stem исходного фото, всегда с расширением .jpg."""
    return next((path for suffix in (".jpg", ".png", ".webp") if (path := folder / f"{image.stem}{suffix}").is_file()), None)


def select_precomputed_images(images: list[Path], folder: Path) -> tuple[list[Path], list[Path]]:
    """Разделяет уже выбранные фото на готовые кропы и отказы нормализации."""
    available = [image for image in images if precomputed_file(image, folder) is not None]
    missing = [image for image in images if precomputed_file(image, folder) is None]
    if missing:
        logger.warning(f"нет готовых кропов: {len(missing)} из {len(images)} выбранных фото; поиск их пропустит, метрики учтут как отказ нормализации: {[image.name for image in missing[:8]]}")
    if not available:
        raise click.UsageError(f"в {folder} нет кропов для выбранных фото")
    return available, missing


class PrecomputedNormalizer:
    """Готовый кроп поиска из S3 и исходное фото для VLM; SAM3 не загружается."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def __call__(self, image: Path) -> list[BottleCrop]:
        path = precomputed_file(image, self.folder)
        if path is None:
            raise FileNotFoundError(f"нет готового кропа для {image} в {self.folder}")
        crop = np.asarray(open_image(path)).copy()
        original = open_image(image)
        original.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        box = np.asarray(original).copy()
        height, width = box.shape[:2]
        whole = [[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]]
        return [BottleCrop(1, 0.0, whole, [], 0.0, crop, {"mode": "precomputed", "source": str(path)}, box, {"mode": "original_fallback"}, original=box)]


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
    elif mode == "precomputed":
        result.timings["normalization_precomputed"] = result.timings.pop("normalization")
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
                np.save(norm_dir / f"{image.stem}_b{item.index}_box_pipeline.npy", item.box_crop)
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
    reranker: NearDuplicateReranker | None,
    norm_cfg: dict | None,
    mode: InputMode,
    batch_size: int,
) -> None:
    """Обрабатывает фото ограниченными пачками и записывает каждый готовый ответ."""
    pending: deque[tuple[Future, PipelineResult, Path, Path, Image.Image | Path]] = deque()

    def finish_one() -> None:
        future, result, image, out, img = pending.popleft()
        result.search, elapsed = future.result()
        result.timings["rerank"] = elapsed
        write_debug_result(image, out, img, result, norm_cfg, searcher, mode)

    def rerank(worker: NearDuplicateReranker, found: list[BottleCandidates | UnmatchedBottle]) -> tuple[list[BottleCandidates | UnmatchedBottle], float]:
        started = time.perf_counter()
        return worker(found), time.perf_counter() - started

    with ThreadPoolExecutor(max_workers=NDR_CONCURRENCY) as executor:
        for start in range(0, len(images), batch_size):
            batch_paths = images[start : start + batch_size]
            outputs = [prepare_output(output_dir, image, mode=mode) for image in batch_paths]
            batch_images = batch_paths if mode in ("cached", "precomputed") else [open_image(image) for image in batch_paths]

            def write_result(
                index: int,
                result: PipelineResult,
                paths: list[Path] = batch_paths,
                dirs: list[Path] = outputs,
                opened: list[Image.Image | Path] = batch_images,
            ) -> None:
                image, out, img = paths[index], dirs[index], opened[index]
                if reranker is None or not result.search:
                    write_debug_result(image, out, img, result, norm_cfg, searcher, mode)
                    return
                worker = reranker.with_trace_dir(out / "rerank") if isinstance(reranker, NearDuplicateReranker) else reranker
                if worker is reranker:
                    worker.trace_dir = out / "rerank"
                pending.append((executor.submit(rerank, worker, result.search), result, image, out, img))
                if len(pending) >= NDR_CONCURRENCY:
                    finish_one()

            pipeline.run_many(batch_images, on_result=write_result, run_rerank=False)
        while pending:
            finish_one()


def expected_slugs(input_path: Path, labels: Path | None) -> dict[str, str]:
    """Разметка теста по имени файла; пустой slug означает вино вне Каталога."""
    manifest = labels or (input_path / "test.csv" if input_path.is_dir() else None)
    if manifest is None or not manifest.is_file():
        return {}
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"image_filename", "slug"}.issubset(reader.fieldnames or []):
            raise click.UsageError(f"в {manifest} нужны колонки image_filename и slug")
        return {
            row["image_filename"].strip(): "" if (slug := (row["slug"] or "").strip()).lower() in {"not_found", "near_duplicate_not_found"} else slug
            for row in reader
        }


def write_metrics(images: list[Path], output_dir: Path, stage: str, truth: dict[str, str], catalog_csv: Path | None = None, missing_normalization: list[Path] | None = None) -> dict:
    """Итог по выбранным фото из result.json; метрики считаются после полного прогона."""
    if truth and stage != "normalization" and (catalog_csv is None or not catalog_csv.is_file()):
        raise click.UsageError("для метрик catalog_only нужен существующий catalog_csv в конфиге поиска")
    known: set[str] | None = None
    if catalog_csv is not None and catalog_csv.is_file():
        with catalog_csv.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if "Slug" not in (reader.fieldnames or []):
                raise click.UsageError(f"в {catalog_csv} нет колонки Slug")
            known = {row["Slug"].strip() for row in reader}
    missing_set = set(missing_normalization or [])
    rows = []
    for image in images:
        report = {"bottles": [], "timings_s": {}} if image in missing_set else json.loads((output_path(output_dir, image) / "result.json").read_text(encoding="utf-8"))
        found = next((b for b in report["bottles"] if b["status"] == "candidates"), None)
        candidates = found["candidates"] if found else []
        expected = truth.get(image.name)
        target = expected if expected is None or known is None or expected in known else ""
        predicted = candidates[0]["slug"] if candidates else ""
        rows.append({
            "image": image.name,
            "expected": expected,
            "target": target,
            "predicted": predicted,
            "rank": next((i for i, c in enumerate(candidates, 1) if c["slug"] == target), None) if target else None,
            "status": "no_normalization" if image in missing_set else "normalized" if stage == "normalization" and any(b["status"] == "normalized" for b in report["bottles"]) else "candidates" if candidates else "no_match",
            "seconds": round(sum(report["timings_s"].values()), 3),
        })
    labeled = [row for row in rows if row["expected"] is not None] if stage != "normalization" else []
    positive = [row for row in labeled if row["target"]]
    negative = [row for row in labeled if not row["target"]]

    def share(values: list[bool]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    def accuracy_at(group: list[dict], k: int, *, include_rejects: bool) -> float | None:
        return share([not row["predicted"] if include_rejects and not row["target"] else row["rank"] is not None and row["rank"] <= k for row in group])

    metrics = {
        "stage": stage,
        "images": len(rows),
        "searched_images": len(rows) - len(missing_set),
        "missing_normalization": len(missing_set),
        "labeled_images": len(labeled),
        "all": {"images": len(labeled), **{f"accuracy_at_{k}": accuracy_at(labeled, k, include_rejects=True) for k in (1, 3, 5)}},
        "catalog_only": {"images": len(positive), **{f"accuracy_at_{k}": accuracy_at(positive, k, include_rejects=False) for k in (1, 3, 5)}},
        "not_found": len(negative),
        "correct_reject": share([not row["predicted"] for row in negative]),
        "false_accept": share([bool(row["predicted"]) for row in negative]),
        "no_match": sum(not row["predicted"] for row in rows),
        "mean_seconds_per_image": round(sum(row["seconds"] for row in rows) / (len(rows) - len(missing_set)), 3) if len(rows) > len(missing_set) else None,
    }
    if stage == "normalization":
        metrics = {"stage": stage, "images": len(rows), "normalized": sum(row["status"] == "normalized" for row in rows), "mean_seconds_per_image": metrics["mean_seconds_per_image"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "per_image.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "expected", "target", "predicted", "rank", "status", "seconds"])
        writer.writeheader()
        writer.writerows(rows)
    if stage == "normalization":
        console.print(f"Нормализация: {metrics['normalized']}/{metrics['images']} фото; среднее {metrics['mean_seconds_per_image']} с")
    else:
        console.print(f"Метрики ({stage}): все {metrics['all']}; с ответом в Каталоге {metrics['catalog_only']}; верные отказы {metrics['correct_reject']}")
    return metrics


def load_saved_search(image: Path, output_dir: Path) -> tuple[dict, list[BottleCandidates | UnmatchedBottle]]:
    """Восстанавливает выдачу поиска из его JSON и сохранённых JPEG, не подключаясь к Qdrant."""
    out = output_path(output_dir, image)
    search_dir = out / "search"
    try:
        report = json.loads((out / "result.json").read_text(encoding="utf-8"))
        saved = json.loads((search_dir / "results.json").read_text(encoding="utf-8"))
        if Path(report["image"]).resolve() != image.resolve():
            raise ValueError(f"result.json относится к другому фото: {report['image']}")
        if not isinstance(saved["queries"], list):
            raise ValueError("search/results.json не содержит список queries")
        bottles = {row["uuid"]: row for row in report["bottles"]}

        def rgb(path: Path) -> np.ndarray:
            with Image.open(path) as stored:
                return np.asarray(stored.convert("RGB")).copy()

        original_image = open_image(image)
        original_image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        original = np.asarray(original_image).copy()
        results: list[BottleCandidates | UnmatchedBottle] = []
        for number, query in enumerate(saved["queries"], 1):
            uuid = query["query"]
            if uuid not in bottles:
                raise ValueError(f"нет бутылки {uuid} в result.json")
            prefix = f"q{number}_query_{uuid[:8]}"
            crop = BottleCrop(
                number, query["bottle_score"], [], [], 0.0,
                rgb(search_dir / f"{prefix}.jpg"), {},
                rgb(search_dir / f"{prefix}_box.jpg"), {}, uuid, original,
            )
            rows = query["candidates"]
            if query["rejected"] is not None:
                if rows:
                    raise ValueError(f"у отказа {uuid} есть кандидаты")
                rejection = query["rejected"]
                results.append(UnmatchedBottle(crop, crop.reject(SearchReason(rejection["reason"]), rejection["detail"]), query.get("selection")))
                continue
            if not rows:
                raise ValueError(f"у запроса {uuid} нет ни кандидатов, ни отказа")
            members = {row["group"]: [item["slug"] for item in rows if item["group"] == row["group"]] for row in rows}
            candidates = []
            for rank, row in enumerate(rows, 1):
                matches = list(search_dir.glob(f"q{number}_{rank:02d}_*.jpg"))
                if len(matches) != 1:
                    raise ValueError(f"ожидался один JPEG кандидата q{number}, ранг {rank}; найдено {len(matches)}")
                # JPEG уже содержит изображение, выбранное поиском; NDR читает его напрямую.
                payload = {FIELD_GROUP_SLUGS: members[row["group"]], "image_crop": "bottle_box"}
                candidates.append(Candidate(row["slug"], row["score"], rgb(matches[0]), crop, row["group"], row["retrieved"], payload))
            results.append(BottleCandidates(crop, candidates, query.get("selection")))
        return report, results
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise click.ClickException(f"не удалось прочитать сохранённый поиск для {image}: {exc}") from exc


def run_saved_ndr(images: list[Path], output_dir: Path, reranker: NearDuplicateReranker) -> None:
    """Запускает только NDR и обновляет итог, не изменяя каталог search/."""
    def process(image: Path) -> dict:
        out = output_path(output_dir, image)
        report, found = load_saved_search(image, output_dir)
        trace_dir = out / "rerank"
        if trace_dir.exists():
            if not trace_dir.resolve().is_relative_to(out):
                raise click.UsageError(f"недопустимая директория результата: {trace_dir}")
            shutil.rmtree(trace_dir)
        worker = reranker.with_trace_dir(trace_dir) if isinstance(reranker, NearDuplicateReranker) else reranker
        if worker is reranker:
            worker.trace_dir = trace_dir
        started = time.perf_counter()
        resolved = worker(found)
        if len(resolved) != len(found) or any(before.crop.uuid != after.crop.uuid for before, after in zip(found, resolved, strict=True)):
            raise click.ClickException(f"NDR вернул неверное число или порядок бутылок для {image}")
        report["timings_s"]["rerank"] = round(time.perf_counter() - started, 3)
        updated = summary(image, PipelineResult([result.crop for result in resolved], resolved))
        by_uuid = {row["uuid"]: row for row in updated["bottles"]}
        report["bottles"] = [by_uuid.get(row["uuid"], row) for row in report["bottles"]]
        (out / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        return report

    with ThreadPoolExecutor(max_workers=NDR_CONCURRENCY) as executor:
        for image, report in zip(images, executor.map(process, images), strict=True):
            print_summary(report)
            logger.info(f"NDR: результаты {output_path(output_dir, image)}")


def input_mode(stage: str, skip_normalization: bool, no_normalization: bool, normalized_dir: Path | None, overrides: tuple[str, ...]) -> InputMode:
    """Два разных обхода SAM3: сохранённые кропы или исходное фото целиком."""
    if sum((skip_normalization, no_normalization, normalized_dir is not None)) > 1:
        raise click.UsageError("--skip-normalization, --no-normalization и --normalized-dir нельзя использовать вместе")
    if stage == "normalization" and (skip_normalization or no_normalization or normalized_dir is not None):
        raise click.UsageError("обход нормализации допустим только с --stage search или full")
    if (skip_normalization or no_normalization or normalized_dir is not None) and overrides:
        raise click.UsageError("--set действует только при запуске SAM3 без обхода нормализации")
    return "cached" if skip_normalization else "raw" if no_normalization else "precomputed" if normalized_dir is not None else "sam3"


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output-dir", type=click.Path(file_okay=False, path_type=Path), required=True, help="Каталог результатов; --stage ndr читает из него сохранённый поиск")
@click.option("--stage", type=click.Choice(["normalization", "search", "ndr", "full"]), default="full", show_default=True, help="Этап: нормализация, поиск, NDR по сохранённому поиску или полный прогон")
@click.option("--limit", type=click.IntRange(min=1), default=None, help="Первые N фото датасета или директории; без лимита — все")
@click.option("--batch-size", type=click.IntRange(min=1), default=2, show_default=True, help="Число фото в пачке поиска; нормализация SAM3 всегда по одному фото")
@click.option("--skip-normalization", is_flag=True, help="Для search/full использовать <output-dir>/<имя>/normalization из прежнего запуска, не загружая SAM3")
@click.option("--no-normalization", is_flag=True, help="Для search/full искать прямо по каждому исходному фото целиком, без SAM3 и сохранённых кропов")
@click.option("--normalized-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None, help="Готовые кропы DINO по stem фото; при --stage ndr определяет фото, прошедшие прежний поиск")
@click.option("--labels", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="CSV с image_filename и slug для итоговых метрик; по умолчанию <input_path>/test.csv")
@click.option("--no-search", is_flag=True, help="Синоним --stage normalization")
@click.option("--search-config", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=DEFAULT_CONFIG, show_default=True, help="Конфиг визуального поиска; его debug_path не используется, разбор пишется в <output-dir>/<имя>/search")
@click.option("-c", "--norm-config", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=NORM_DIR / "normalize.toml", show_default=True, help="Конфиг нормализации")
@click.option("--set", "overrides", multiple=True, metavar="секция.ключ=значение", help="Переопределить параметр конфига нормализации")
def main(input_path: Path, output_dir: Path, stage: str, limit: int | None, batch_size: int, skip_normalization: bool, no_normalization: bool, normalized_dir: Path | None, labels: Path | None, no_search: bool, search_config: Path, norm_config: Path, overrides: tuple[str, ...]) -> None:
    """Прогнать выбранные фото до этапа или выполнить NDR по сохранённой выдаче поиска.

    <output-dir>/<имя файла>/normalization — кропы годных бутылок, маски и json как у CLI нормализации, плюс markup.json со всеми
    бутылками и причинами отказов; search — разбор поиска в формате debug_path: кроп запроса, картинки кандидатов с косинусом
    и группой в имени, results.json; result.json — итог по каждой бутылке и время шагов. Устройства — NORMALIZER_DEV и VIS_SEARCHER_DEV.
    """
    load_dotenv(ROOT / ".env", override=False)
    if no_search:
        stage = "normalization"
    if stage == "ndr" and (skip_normalization or no_normalization or overrides):
        raise click.UsageError("--stage ndr читает сохранённый поиск; --skip-normalization, --no-normalization и --set здесь не нужны")
    mode = input_mode(stage, skip_normalization, no_normalization, normalized_dir, overrides)
    selected_images = select_images(input_path, limit)
    images, missing_normalization = select_precomputed_images(selected_images, normalized_dir) if normalized_dir is not None else (selected_images, [])
    if stage == "ndr":
        cfg = load_search_config(search_config)
        reranker = NearDuplicateReranker.from_search_config(cfg)
        logger.info(f"этап ndr; изображений {len(images)}; поиск читается из {output_dir}")
        run_saved_ndr(images, output_dir, reranker)
        write_metrics(selected_images, output_dir, stage, expected_slugs(input_path, labels), cfg.catalog_csv, missing_normalization)
        return
    if mode == "cached":
        for image in images:
            marker = output_path(output_dir, image) / "normalization" / "markup.json"
            if not marker.is_file():
                raise click.UsageError(f"нет сохранённой нормализации для {image}: {marker}")
    norm_cfg = load_norm_config(norm_config, list(overrides)) if mode == "sam3" else None
    normalizer = Normalizer(norm_cfg, norm_config.resolve().parent) if mode == "sam3" else CachedNormalizer(output_dir, include_original=stage == "full") if mode == "cached" else PrecomputedNormalizer(normalized_dir) if mode == "precomputed" else RawImageNormalizer()
    searcher = None
    reranker = None
    catalog_csv = None
    if stage != "normalization":
        cfg = load_search_config(search_config)
        catalog_csv = getattr(cfg, "catalog_csv", None)
        cfg.debug_path = None
        try:
            searcher = VisSearcher(cfg)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        if stage == "full":
            reranker = NearDuplicateReranker.from_search_config(cfg)
    pipeline = Pipeline(normalizer, searcher, reranker, enable_rerank=stage == "full")
    search_batch_size = batch_size if searcher is not None else 1
    logger.info(f"этап {stage}; изображений {len(images)}; фото в пачке {search_batch_size}; вход {mode}")
    run_debug_batches(images, output_dir, pipeline, searcher, reranker, norm_cfg, mode, search_batch_size)
    write_metrics(selected_images, output_dir, stage, expected_slugs(input_path, labels), catalog_csv, missing_normalization)


if __name__ == "__main__":
    main()
