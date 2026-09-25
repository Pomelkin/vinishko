"""Rank catalog wines for the unlabeled field photos using the exported ONNX model.

Run from the repository root with ``.venv/Scripts/python scripts/retrieve_new_test.py``.
The checkpoint files in ``data/.new_test_retrieval_cache`` allow interrupted CPU runs
to resume. Delete that directory to rebuild the embeddings from scratch.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/strapi/catalog_dataset.csv"
IMAGES = ROOT / "data/strapi/img"
QUERIES = ROOT / "data/new_test"
MODEL_DIR = ROOT / "vis_searcher/weights/dinov3_vitl16_512"
CACHE = ROOT / "data/.new_test_retrieval_cache"
OUTPUT = ROOT / "data/new_test_top5.csv"


def load_rgb(path: Path, pad: tuple[int, int, int], *, alpha_crop: bool) -> np.ndarray:
    """Read by image signature, apply orientation, and flatten transparency."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
        if alpha_crop:
            alpha = image.getchannel("A")
            if alpha.getextrema()[0] < 250:
                bounds = alpha.point(lambda value: 255 if value > 5 else 0).getbbox()
                if bounds is not None:
                    image = image.crop(bounds)
        background = Image.new("RGBA", image.size, (*pad, 255))
        return np.asarray(Image.alpha_composite(background, image).convert("RGB"))


def model_input(rgb: np.ndarray, size: tuple[int, int], pad: tuple[int, int, int]) -> np.ndarray:
    """Keep proportions, center, and send unnormalized RGB 0..255 float32."""
    height, width = rgb.shape[:2]
    target_h, target_w = size
    scale = min(target_h / height, target_w / width)
    new_h, new_w = max(1, round(height * scale)), max(1, round(width * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=interpolation)
    canvas = np.empty((target_h, target_w, 3), dtype=np.uint8)
    canvas[:] = pad
    top, left = (target_h - new_h) // 2, (target_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)


def center_crop(rgb: np.ndarray) -> np.ndarray:
    """Focus a second query view on the central target bottle."""
    height, width = rgb.shape[:2]
    return rgb[round(height * 0.06) : round(height * 0.94), round(width * 0.17) : round(width * 0.83)]


def fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    with CATALOG.open(encoding="utf-8-sig", newline="") as stream:
        catalog = list(csv.DictReader(stream))
    catalog = [row for row in catalog if row["Название фото"] and (IMAGES / row["Название фото"]).is_file()]
    queries = sorted(QUERIES.glob("*.webp"))
    if len({row["Slug"] for row in catalog}) != len(catalog):
        raise ValueError("Catalog image rows have duplicate slugs")

    prep = json.loads((MODEL_DIR / "preprocess.json").read_text(encoding="utf-8"))
    size = tuple(prep["input_size"])
    pad = tuple(prep["pad_color"])
    key = fingerprint([CATALOG, MODEL_DIR / "preprocess.json", MODEL_DIR / "model.onnx"])
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = CACHE / "manifest.json"
    expected = {"fingerprint": key, "catalog_slugs": [row["Slug"] for row in catalog], "queries": [p.name for p in queries]}
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != expected:
        raise ValueError("Retrieval cache belongs to different inputs; delete it before rebuilding")
    if not manifest.exists():
        manifest.write_text(json.dumps(expected, ensure_ascii=False), encoding="utf-8")

    core = ov.Core()
    model = core.read_model(MODEL_DIR / "model.onnx")
    compiled = core.compile_model(model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
    if model.outputs[0].get_partial_shape()[1].is_static:
        dims = int(model.outputs[0].get_partial_shape()[1].get_length())
    else:
        raise ValueError("The embedding dimension must be static")

    catalog_vectors = np.lib.format.open_memmap(CACHE / "catalog.npy", mode="r+" if (CACHE / "catalog.npy").exists() else "w+", dtype=np.float32, shape=(len(catalog), dims))
    catalog_done = np.lib.format.open_memmap(CACHE / "catalog_done.npy", mode="r+" if (CACHE / "catalog_done.npy").exists() else "w+", dtype=np.bool_, shape=(len(catalog),))
    query_vectors = np.lib.format.open_memmap(CACHE / "queries.npy", mode="r+" if (CACHE / "queries.npy").exists() else "w+", dtype=np.float32, shape=(len(queries), 2, dims))
    query_done = np.lib.format.open_memmap(CACHE / "queries_done.npy", mode="r+" if (CACHE / "queries_done.npy").exists() else "w+", dtype=np.bool_, shape=(len(queries), 2))

    by_image: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(catalog):
        by_image[row["Название фото"]].append(index)

    def infer(rgb: np.ndarray) -> np.ndarray:
        result = np.asarray(compiled(model_input(rgb, size, pad))[compiled.output(0)][0], dtype=np.float32)
        norm = float(np.linalg.norm(result))
        if not np.isfinite(norm) or norm < 0.9:
            raise ValueError(f"Invalid ONNX embedding norm: {norm}")
        return result / norm

    for count, (name, indices) in enumerate(by_image.items(), 1):
        if not all(catalog_done[index] for index in indices):
            vector = infer(load_rgb(IMAGES / name, pad, alpha_crop=True))
            for index in indices:
                catalog_vectors[index] = vector
                catalog_done[index] = True
        if count % 25 == 0 or count == len(by_image):
            catalog_vectors.flush()
            catalog_done.flush()
            print(f"Catalog {count}/{len(by_image)}", flush=True)

    for index, path in enumerate(queries):
        if not query_done[index].all():
            full = load_rgb(path, pad, alpha_crop=False)
            for view, rgb in enumerate((full, center_crop(full))):
                if not query_done[index, view]:
                    query_vectors[index, view] = infer(rgb)
                    query_done[index, view] = True
        if (index + 1) % 10 == 0:
            query_vectors.flush()
            query_done.flush()
            print(f"Queries {index + 1}/{len(queries)}", flush=True)

    fields = ["query_file", "rank", "slug", "cosine_similarity", "cosine_distance", "best_view", "catalog_image", "wine_name", "winery", "category", "grapes", "vintage", "sugar"]
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, path in enumerate(queries):
            scores = np.asarray(query_vectors[index] @ catalog_vectors.T)
            best_view = scores.argmax(axis=0)
            best_score = scores.max(axis=0)
            nearest = np.argsort(-best_score, kind="stable")[:5]
            for rank, item in enumerate(nearest, 1):
                row = catalog[int(item)]
                score = float(best_score[item])
                writer.writerow({
                    "query_file": path.name,
                    "rank": rank,
                    "slug": row["Slug"],
                    "cosine_similarity": f"{score:.6f}",
                    "cosine_distance": f"{1 - score:.6f}",
                    "best_view": ("full", "center")[int(best_view[item])],
                    "catalog_image": row["Название фото"],
                    "wine_name": row["Название вина"],
                    "winery": row["Винодельня"],
                    "category": row["Категория"],
                    "grapes": row["Сорт винограда"],
                    "vintage": row["Винтаж"],
                    "sugar": row["Сахар"],
                })
    print(f"Saved {len(queries) * 5} candidates to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
