"""Запуск из корня: python -m vinishko.pipeline-siglip.debug. Параметры ниже."""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from vinishko.pipeline.structs import UnmatchedBottle

from .pipeline import Pipeline, PipelineResult
from .steps.vis_searcher import VisSearcher

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "datasets/local/test"
OUTPUT_DIR = ROOT / "datasets/local/runs_siglip"
INPUT_MODE = "images_crop_box"  # raw | images_crop_box
BOX_RUNS_DIR = ROOT / "datasets/local/runs_vanilla"
LIMIT = None  # None — все фото
BATCH_SIZE = 2
DROP_NA = False
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def select_images(input_path: Path, limit: int | None, drop_na: bool = False) -> list[Path]:
    """Один файл, test.csv/catalog.csv или папка изображений, как в pipeline.debug."""
    if limit is not None and limit < 1:
        raise ValueError("LIMIT должен быть положительным или None")
    if input_path.is_file():
        images = [input_path]
    else:
        manifest = next((input_path / name for name in ("test.csv", "catalog.csv") if (input_path / name).is_file()), None)
        if manifest is not None:
            with manifest.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if "image_filename" not in (reader.fieldnames or []):
                    raise ValueError(f"в {manifest} нет image_filename")
                images = [input_path / "images" / name for row in reader
                          if not (drop_na and any(not (value or "").strip() for value in row.values()))
                          if (name := (row["image_filename"] or "").strip())]
        else:
            folder = input_path / "images" if (input_path / "images").is_dir() else input_path
            images = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=lambda p: p.name.casefold())
    images = images[:limit]
    if not images:
        raise ValueError(f"в {input_path} нет изображений")
    if missing := [str(p) for p in images if not p.is_file()]:
        raise FileNotFoundError(f"не найдены изображения: {missing[:3]}")
    return images


def summary(image: Path, result: PipelineResult, input_mode: str = "raw") -> dict:
    bottles = []
    for answer in result.search:
        if isinstance(answer, UnmatchedBottle):
            bottles.append({"uuid": answer.crop.uuid, "status": "rejected", "step": "search",
                            "reason": answer.rejected.reason, "message": answer.rejected.message,
                            "score": answer.rejected.score, "selection": answer.selection})
        else:
            bottles.append({"uuid": answer.crop.uuid, "status": "candidates", "bottle_score": answer.crop.score,
                            "candidates": [{"slug": c.slug, "group": c.group, "score": c.score,
                                            "retrieved": c.retrieved} for c in answer.candidates],
                            "selection": answer.selection})
    return {"image": str(image), "input_mode": input_mode, "timings_s": result.timings, "bottles": bottles}


def box_input(image: Path, runs_dir: Path) -> Path | None:
    """Готовый render_bottle_box из сохранённой нормализации."""
    path = runs_dir / image.stem / "normalization" / f"{image.stem}_b1_box.jpg"
    return path if path.is_file() else None


def main() -> None:
    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE должен быть положительным")
    if INPUT_MODE not in {"raw", "images_crop_box"}:
        raise ValueError(f"неизвестный INPUT_MODE: {INPUT_MODE}")
    images = select_images(INPUT_PATH, LIMIT, DROP_NA)
    run_dir = OUTPUT_DIR / datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S_%f")
    searcher = VisSearcher()
    try:
        pipeline = Pipeline(searcher)
        for start in range(0, len(images), BATCH_SIZE):
            paths = images[start:start + BATCH_SIZE]
            selected = []
            for index, path in enumerate(paths, start + 1):
                source = path if INPUT_MODE == "raw" else box_input(path, BOX_RUNS_DIR)
                if source is None:
                    out = run_dir / f"{index:05d}_{path.stem}"
                    out.mkdir(parents=True, exist_ok=True)
                    (out / "result.json").write_text(
                        json.dumps({"image": str(path), "input_mode": INPUT_MODE, "timings_s": {}, "bottles": []}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    print(f"{path.name}: нет images_crop_box → {out}")
                else:
                    selected.append((index, path, source))
            for (index, path, _), result in zip(selected, pipeline.run_many([source for _, _, source in selected]), strict=True):
                out = run_dir / f"{index:05d}_{path.stem}"
                searcher.dump(result.search, out / "search")
                report = summary(path, result, INPUT_MODE)
                (out / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"{path.name}: {report['bottles'][0]['status']} → {out}")
    finally:
        searcher.close()


if __name__ == "__main__":
    main()
