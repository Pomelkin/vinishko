#!/usr/bin/env python3
"""Legacy-скрининг near-duplicate-кандидатов на уровне файлов Эталонов.

Считает текстовые группы, общий файл и визуальную близость, затем локализует пиксельную
разницу. Эти метрики не определяют идентичность вина: свет, кроп и новый рендер дают большую
разницу даже для того же объекта. Итоговую object-level классификацию выполняет
``semantic_near_duplicates.py``.

    python3 scripts/near_duplicates.py              # текстовый отчёт
    python3 scripts/near_duplicates.py --json       # машинный формат
    python3 scripts/near_duplicates.py --no-visual  # только текстовые группы, без Pillow/numpy

Служит генератором кандидатов и модулем для ``collect_near_duplicates.py``. Итоговые цифры
в ``docs/NEAR_DUPLICATES.md`` получены ``semantic_near_duplicates.py``.

Требуется Pillow и numpy (`python3 -m pip install --cert /etc/ssl/cert.pem numpy`)
для визуальной части; текстовая работает без них.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import analyze_data as ad  # noqa: E402  — общая загрузка Каталога и связывание с Медиа-дампом

IMG_DIR = ad.IMG_DIR

# --- пороги -----------------------------------------------------------------
# Откалиброваны на самих данных (см. docs/NEAR_DUPLICATES.md):
# копии одного и того же рендера, пересохранённые Strapi, дают maxdiff <= 0.12;
# минимальный maxdiff у пары РАЗНЫХ позиций Каталога — 0.356. Разрыв в 3 раза,
# поэтому 0.20 разделяет «тот же рендер» и «есть хоть одна различающая деталь».
RECODE_NOISE_MAXDIFF = 0.20
PREVIEW_CANDIDATE_RMSE = 0.06   # кандидаты на точное сравнение (baseline p1 по всем парам — 0.179)
DIFF_PIXEL_THRESHOLD = 0.10     # пиксель считается различающимся
BAND_THRESHOLD = 0.15           # порог для выделения полос различий
MICRO_DIFF_FRAC = 0.005         # < 0.5% площади бутылки — «различие размером с надпись»
SMALL_DIFF_FRAC = 0.02          # < 2% площади — «различие размером со строку текста»
BOTTLE_WIDTH = 640              # ширина нормализованной бутылки при точном сравнении
PREVIEW_SIZE = (128, 256)       # превью для попарного скрининга (ширина, высота)

SUGAR_TOKENS = ["ekstra-bryut", "bryut", "polusladkoe", "polusuhoe", "sladkoe", "suhoe"]
COLOR_TOKENS = ["krasnoe", "beloe", "rozovoe", "oranzhevoe"]


# --- Каталог ----------------------------------------------------------------

def load_catalog_with_images() -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Каталог (slug -> строка) и привязанные Эталоны (slug -> файлы в Медиа-дампе)."""
    catalog = ad.load_catalog()
    orig = ad.original_names()
    link = ad.link_catalog_to_images(catalog, orig)
    slug_files: dict[str, set[str]] = collections.defaultdict(set)
    for source in (link["by_photo_name"], link["by_slug_name"]):
        for slug, files in source.items():
            slug_files[slug].update(files)
    return catalog, {slug: sorted(files) for slug, files in slug_files.items()}


def norm_name(name: str) -> str:
    """Название без года, объёма и пунктуации — ключ серии."""
    s = name.strip().lower().replace("ё", "е")
    s = re.sub(r"\b(19|20)\d{2}\b", " ", s)
    s = re.sub(r"\b0[.,]\d+\s*(л|l)\b", " ", s)
    s = re.sub(r"[^\w\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_attrs(slug: str, row: dict) -> dict:
    """Атрибуты позиции: часть из полей Каталога, часть распарсена из структуры slug."""
    sugar = next((t for t in SUGAR_TOKENS if re.search(rf"(^|-){t}(-|$)", slug)), None)
    color = next((t for t in COLOR_TOKENS if re.search(rf"(^|-){t}(-|$)", slug)), None)
    abv = re.search(r"-(\d{2,4})$", slug)
    year_name = re.search(r"\b(19|20)\d{2}\b", row["Название вина"])
    year_slug = re.search(r"-((?:19|20)\d{2})(?:-|$)", slug)
    return {
        "sugar": sugar,
        "color": color,
        "abv": abv.group(1) if abv else None,
        "year": year_name.group(0) if year_name else (year_slug.group(1) if year_slug else None),
        "category": row["Категория"].strip(),
        "grape": row["Сорт винограда"].strip().lower(),
        "winery": row["Винодельня"].strip(),
        "name": row["Название вина"].strip(),
        "manual_tail": bool(re.search(r"-\d$", slug)),
    }


DISCRIMINATORS = [
    ("цвет вина", "category"),
    ("сорт", "grape"),
    ("сахар", "sugar"),
    ("год", "year"),
    ("крепость", "abv"),
]


def discriminators(slugs: list[str], catalog: dict) -> list[str]:
    """Какие атрибуты различают позиции внутри группы (пустой список = нечем развести)."""
    attrs = {s: parse_attrs(s, catalog[s]) for s in slugs}
    return [label for label, field in DISCRIMINATORS if len({attrs[s][field] for s in slugs}) > 1]


def series_groups(catalog: dict) -> dict[tuple[str, str], list[str]]:
    """Группы «винодельня + название без года/объёма» — одна серия у одного производителя."""
    groups: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for slug, row in catalog.items():
        groups[(row["Винодельня"].strip(), norm_name(row["Название вина"]))].append(slug)
    return {key: sorted(slugs) for key, slugs in groups.items() if len(slugs) > 1}


def name_collisions(catalog: dict) -> dict[str, list[str]]:
    """Одно название вина у разных виноделен — ловушка для текстового канала."""
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for slug, row in catalog.items():
        groups[norm_name(row["Название вина"])].append(slug)
    out = {}
    for name, slugs in groups.items():
        if len({catalog[s]["Винодельня"].strip() for s in slugs}) > 1:
            out[name] = sorted(slugs)
    return out


# --- изображения ------------------------------------------------------------

class ImageBank:
    """Нормализованные Эталоны: кроп по альфе, белый фон, фиксированная ширина."""

    def __init__(self, files: list[str]):
        from PIL import Image
        import numpy as np
        Image.MAX_IMAGE_PIXELS = None
        self._Image, self._np = Image, np
        self.files = sorted(files)
        self.sha: dict[str, str] = {}
        self._preview: dict[str, object] = {}
        self._full: dict[str, object] = {}

    def digest(self, name: str) -> str:
        if name not in self.sha:
            self.sha[name] = hashlib.sha256((IMG_DIR / name).read_bytes()).hexdigest()
        return self.sha[name]

    def normalized(self, name: str, width: int):
        """PIL.Image: бутылка обрезана по непрозрачным пикселям и положена на белый фон."""
        Image = self._Image
        with Image.open(IMG_DIR / name) as im:
            im.load()
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                im = im.convert("RGBA")
                box = im.getchannel("A").getbbox()
                if box:
                    im = im.crop(box)
                canvas = Image.new("RGB", im.size, (255, 255, 255))
                canvas.paste(im, mask=im.getchannel("A"))
                im = canvas
            else:
                im = im.convert("RGB")
            height = max(1, round(im.height * width / im.width))
            return im.resize((width, height), Image.LANCZOS)

    def preview(self, name: str):
        if name not in self._preview:
            np = self._np
            im = self.normalized(name, PREVIEW_SIZE[0]).resize(PREVIEW_SIZE, self._Image.LANCZOS)
            gray = (np.asarray(im, np.float32) / 255).mean(axis=2)
            self._preview[name] = gray.reshape(32, gray.shape[0] // 32, 32, gray.shape[1] // 32).mean(axis=(1, 3))
        return self._preview[name]

    def full(self, name: str):
        """Нормализованная бутылка как uint8-массив (float32 раздул бы кэш в четыре раза)."""
        if name not in self._full:
            if len(self._full) > 160:          # ~5 МБ на файл: оригиналы бывают 1603x4810
                self._full.clear()
            self._full[name] = self._np.asarray(self.normalized(name, BOTTLE_WIDTH), self._np.uint8)
        return self._full[name]

    def preview_distances(self):
        """Матрица RMSE между превью всех Эталонов (диагональ = inf)."""
        np = self._np
        vectors = np.stack([self.preview(f).ravel() for f in self.files])
        squares = (vectors ** 2).sum(1)
        d2 = squares[:, None] + squares[None, :] - 2 * vectors @ vectors.T
        np.fill_diagonal(d2, np.inf)
        return np.sqrt(np.maximum(d2, 0) / vectors.shape[1])


def aligned_difference(bank: ImageBank, name_a: str, name_b: str, max_shift: int = 12):
    """Карта попиксельной разницы после подбора лучшего сдвига (рендеры кадрированы по-разному).

    Сдвиг ищется в два прохода: грубо на копии, уменьшенной вчетверо, потом уточнение
    в полном разрешении — перебор всех сдвигов сразу на 640-пиксельной бутылке слишком дорог.
    """
    np = bank._np
    A = bank.full(name_a).astype(np.float32) / 255
    B = bank.full(name_b).astype(np.float32) / 255
    height = min(A.shape[0], B.shape[0])
    A, B = A[:height], B[:height]

    def error_at(a_src, b_src, dy, dx):
        h_max = a_src.shape[0]
        a = a_src[max(0, dy):h_max + min(0, dy), max(0, dx):a_src.shape[1] + min(0, dx)]
        b = b_src[max(0, -dy):h_max + min(0, -dy), max(0, -dx):b_src.shape[1] + min(0, -dx)]
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        return float(np.abs(a[:h, :w] - b[:h, :w]).mean()), a[:h, :w], b[:h, :w]

    step = 4
    small_a = A[:height // step * step].reshape(height // step, step, -1, step, 3).mean(axis=(1, 3))
    small_b = B[:height // step * step].reshape(height // step, step, -1, step, 3).mean(axis=(1, 3))
    coarse = min(
        ((error_at(small_a, small_b, dy, dx)[0], dy, dx)
         for dy in range(-max_shift // step, max_shift // step + 1)
         for dx in range(-max_shift // step, max_shift // step + 1)),
        key=lambda item: item[0])
    base_y, base_x = coarse[1] * step, coarse[2] * step

    best = None
    for dy in range(base_y - step, base_y + step + 1, 2):
        for dx in range(base_x - step, base_x + step + 1, 2):
            error, a, b = error_at(A, B, dy, dx)
            if best is None or error < best[0]:
                best = (error, np.abs(a - b).mean(axis=2), (dy, dx))
    return best[1], best[2]


def diff_bands(diff_map, threshold: float = BAND_THRESHOLD, gap: int = 8):
    """Горизонтальные полосы различий: (y0, y1, x0, x1, площадь в px) сверху вниз."""
    import numpy as np
    mask = diff_map > threshold
    rows = np.where(mask.any(1))[0]
    bands = []
    if not len(rows):
        return bands
    start = prev = rows[0]
    for row in list(rows[1:]) + [10 ** 9]:
        if row - prev > gap:
            chunk = mask[start:prev + 1]
            cols = np.where(chunk.any(0))[0]
            bands.append((int(start), int(prev), int(cols.min()), int(cols.max()), int(chunk.sum())))
            start = row
        prev = row
    return bands


def compare_pair(bank: ImageBank, file_a: str, file_b: str) -> dict:
    """Метрики различия двух Эталонов плюс локализация различающих деталей."""
    if bank.digest(file_a) == bank.digest(file_b):
        return {"relation": "same_file", "max_diff": 0.0, "diff_frac": 0.0, "bands": []}
    diff_map, shift = aligned_difference(bank, file_a, file_b)
    height, width = diff_map.shape
    frac = float((diff_map > DIFF_PIXEL_THRESHOLD).mean())
    max_diff = float(diff_map.max())
    bands = [
        {
            "y0_pct": round(y0 / height, 3), "y1_pct": round(y1 / height, 3),
            "height_px": y1 - y0 + 1, "width_px": x1 - x0 + 1, "area_px": area,
        }
        for y0, y1, x0, x1, area in diff_bands(diff_map)
        if area >= width * 0.001                      # отсечь точечный шум компрессии
    ]
    if max_diff < RECODE_NOISE_MAXDIFF:
        relation = "pixel_identical"
    elif frac < MICRO_DIFF_FRAC:
        relation = "micro_diff"
    elif frac < SMALL_DIFF_FRAC:
        relation = "small_diff"
    else:
        relation = "visible_diff"
    return {
        "relation": relation, "max_diff": round(max_diff, 3), "diff_frac": round(frac, 5),
        "mean_diff": round(float(diff_map.mean()), 5), "shift": shift,
        "bottle_px": [width, height],
        "bands": sorted(bands, key=lambda b: -b["area_px"])[:6],
    }


RELATION_LABELS = {
    "same_file": "одни байты рендера — об идентичности физического объекта не свидетельствует",
    "pixel_identical": "разные файлы, но ни одной различающей детали",
    "micro_diff": "различие < 0.5% площади бутылки (надпись, год)",
    "small_diff": "различие < 2% площади (строка текста, поясок этикетки)",
    "visible_diff": "различие видно сразу",
}


# --- сборка групп -----------------------------------------------------------

def build_groups(with_visual: bool = True, max_visual_pairs: int = 600) -> dict:
    """Единая структура групп near-duplicates — её же использует collect_near_duplicates.py."""
    catalog, slug_files = load_catalog_with_images()
    groups = []

    bank = None
    preview_rmse = None
    file_index: dict[str, int] = {}
    if with_visual:
        files = sorted({f for fs in slug_files.values() for f in fs})
        bank = ImageBank(files)
        preview_rmse = bank.preview_distances()
        file_index = {f: i for i, f in enumerate(bank.files)}

    # --- A. один Эталон на несколько позиций
    by_digest: dict[str, set[str]] = collections.defaultdict(set)
    digest_files: dict[str, set[str]] = collections.defaultdict(set)
    if bank is not None:
        for slug, files in slug_files.items():
            for name in files:
                by_digest[bank.digest(name)].add(slug)
                digest_files[bank.digest(name)].add(name)
    else:                                              # без Pillow — по «Название фото» из Каталога
        for slug, row in catalog.items():
            by_digest[row["Название фото"].strip()].add(slug)
    seen_a: set[frozenset] = set()
    for digest, slugs in by_digest.items():
        if len(slugs) < 2 or frozenset(slugs) in seen_a:
            continue
        seen_a.add(frozenset(slugs))
        groups.append({
            "type": "A_one_image_many_slugs",
            "key": sorted(slugs)[0],
            "slugs": sorted(slugs),
            "files": sorted(digest_files.get(digest, [])),
            "discriminators": discriminators(sorted(slugs), catalog),
            "wineries": sorted({catalog[s]["Винодельня"].strip() for s in slugs}),
            "relation": "same_file",
        })

    # --- B. одна серия у одного производителя
    # пары из A уже описаны как отдельные группы — в C их повторно не заводим
    covered_pairs: set[frozenset] = {
        frozenset(pair) for group in groups
        for pair in itertools.combinations(group["slugs"], 2)
    }
    for (winery, name), slugs in sorted(series_groups(catalog).items()):
        groups.append({
            "type": "B_same_series",
            "key": f"{winery} / {name}",
            "slugs": slugs,
            "discriminators": discriminators(slugs, catalog),
            "wineries": [winery],
            "with_reference": [s for s in slugs if s in slug_files],
        })
        for pair in itertools.combinations(slugs, 2):
            covered_pairs.add(frozenset(pair))

    # --- C. визуально близкие Эталоны, не покрытые серией
    if bank is not None:
        import numpy as np
        upper = np.triu_indices(len(bank.files), 1)
        values = preview_rmse[upper]
        file_slugs: dict[str, list[str]] = collections.defaultdict(list)
        for slug, files in slug_files.items():
            for name in files:
                file_slugs[name].append(slug)
        order = [k for k in np.argsort(values) if values[k] < PREVIEW_CANDIDATE_RMSE]
        visual = []
        for k in order[:max_visual_pairs]:
            file_a, file_b = bank.files[upper[0][k]], bank.files[upper[1][k]]
            slugs_a, slugs_b = set(file_slugs[file_a]), set(file_slugs[file_b])
            if slugs_a == slugs_b:                      # копии рендера одной и той же позиции
                continue
            for slug_a, slug_b in itertools.product(sorted(slugs_a), sorted(slugs_b)):
                if slug_a == slug_b or frozenset((slug_a, slug_b)) in covered_pairs:
                    continue
                covered_pairs.add(frozenset((slug_a, slug_b)))
                visual.append((float(values[k]), slug_a, slug_b, file_a, file_b))
        for preview, slug_a, slug_b, file_a, file_b in visual:
            metrics = compare_pair(bank, file_a, file_b)
            groups.append({
                "type": "C_visually_close",
                "key": f"{slug_a}|{slug_b}",
                "slugs": [slug_a, slug_b],
                "files": [file_a, file_b],
                "preview_rmse": round(preview, 4),
                "discriminators": discriminators([slug_a, slug_b], catalog),
                "wineries": sorted({catalog[s]["Винодельня"].strip() for s in (slug_a, slug_b)}),
                **{k: v for k, v in metrics.items()},
            })

    # метрики для пар внутри серий, у которых есть два Эталона
    if bank is not None:
        for group in groups:
            if group["type"] != "B_same_series":
                continue
            pairs = []
            have = group["with_reference"]
            for slug_a, slug_b in itertools.combinations(have, 2):
                file_a, file_b = slug_files[slug_a][0], slug_files[slug_b][0]
                metrics = compare_pair(bank, file_a, file_b)
                metrics.update({"slugs": [slug_a, slug_b], "files": [file_a, file_b],
                                "preview_rmse": round(float(preview_rmse[file_index[file_a],
                                                                         file_index[file_b]]), 4)
                                if file_a != file_b else 0.0})
                pairs.append(metrics)
            group["pairs"] = pairs

    return {"catalog": catalog, "slug_files": slug_files, "groups": groups, "bank": bank,
            "visual": bank is not None}


# --- отчёт ------------------------------------------------------------------

def build_report(state: dict) -> dict:
    catalog, slug_files, groups = state["catalog"], state["slug_files"], state["groups"]
    by_type = collections.defaultdict(list)
    for group in groups:
        by_type[group["type"]].append(group)

    shared = by_type["A_one_image_many_slugs"]
    series = by_type["B_same_series"]
    visual = by_type["C_visually_close"]

    series_slugs = {s for g in series for s in g["slugs"]}
    shared_slugs = {s for g in shared for s in g["slugs"]}
    disc_counter = collections.Counter(
        " + ".join(g["discriminators"]) if g["discriminators"] else "НИЧЕГО" for g in series)

    pairs = [p for g in series for p in g.get("pairs", [])] + [
        {k: g[k] for k in ("relation", "max_diff", "diff_frac", "slugs") if k in g} for g in visual]
    relation_counter = collections.Counter(p["relation"] for p in pairs)

    cross_winery_visual = [g for g in visual if len(g["wineries"]) > 1]
    real_pairs = [p for p in pairs if p["relation"] != "same_file" and "max_diff" in p]

    collisions = name_collisions(catalog)
    return {
        "catalog_size": len(catalog),
        "slugs_with_reference": len(slug_files),
        "A_one_image_many_slugs": {
            "groups": len(shared),
            "slugs": len(shared_slugs),
            "same_winery": sum(1 for g in shared if len(g["wineries"]) == 1),
            "cross_winery": sum(1 for g in shared if len(g["wineries"]) > 1),
            "different_category": sum(
                1 for g in shared if len({catalog[s]["Категория"].strip() for s in g["slugs"]}) > 1),
        },
        "B_same_series": {
            "groups": len(series),
            "slugs": len(series_slugs),
            "groups_with_two_references": sum(1 for g in series if len(g["with_reference"]) >= 2),
            "groups_without_reference": sum(1 for g in series if not g["with_reference"]),
            "discriminators": dict(disc_counter.most_common()),
            "groups_without_discriminator": disc_counter.get("НИЧЕГО", 0),
        },
        "C_visually_close": {
            "pairs": len(visual),
            "cross_winery": len(cross_winery_visual),
        },
        "pair_relations": dict(relation_counter),
        "min_max_diff_between_positions": (
            round(min(p["max_diff"] for p in real_pairs), 3) if real_pairs else None),
        "name_collisions": {
            "names": len(collisions),
            "slugs": len({s for v in collisions.values() for s in v}),
        },
    }


def print_report(state: dict, report: dict) -> None:
    def section(title):
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")

    groups = state["groups"]
    catalog = state["catalog"]
    by_type = collections.defaultdict(list)
    for group in groups:
        by_type[group["type"]].append(group)

    section("ОБЩЕЕ")
    print(f"  позиций в Каталоге                {report['catalog_size']}")
    print(f"  из них с Эталоном                 {report['slugs_with_reference']}")

    section("A. ОДИН ФАЙЛ НА НЕСКОЛЬКО ПОЗИЦИЙ — алиас либо коллизия данных")
    a = report["A_one_image_many_slugs"]
    print(f"  групп {a['groups']}, позиций {a['slugs']}; одна винодельня {a['same_winery']}, "
          f"разные {a['cross_winery']}, разный цвет вина {a['different_category']}")
    if not state.get("visual"):
        print("  ВНИМАНИЕ: --no-visual считает эту секцию по полю «Название фото», а не по\n"
              "  содержимому файлов. Байт-идентичные файлы с разными именами не видны, поэтому\n"
              "  групп получается меньше, чем при полном прогоне (13/26 против 20/40).")
    for group in sorted(by_type["A_one_image_many_slugs"], key=lambda g: g["slugs"]):
        marks = ", ".join(group["discriminators"]) or "нечем развести по атрибутам"
        print(f"   * {group['slugs']}")
        print(f"     {' / '.join(group['wineries'])} | различает: {marks}")

    section("B. ОДНА СЕРИЯ У ОДНОГО ПРОИЗВОДИТЕЛЯ")
    b = report["B_same_series"]
    print(f"  групп {b['groups']}, позиций {b['slugs']}")
    print(f"  с двумя и более Эталонами {b['groups_with_two_references']}, "
          f"без единого Эталона {b['groups_without_reference']}")
    print("\n  чем различаются позиции внутри группы:")
    for label, count in b["discriminators"].items():
        print(f"    {count:3d}  {label}")

    section("C. ВИЗУАЛЬНО БЛИЗКИЕ ЭТАЛОНЫ ВНЕ ОДНОЙ СЕРИИ")
    c = report["C_visually_close"]
    print(f"  пар {c['pairs']}, из них у разных виноделен {c['cross_winery']}")

    section("НАСКОЛЬКО ПОХОЖИ ПАРЫ")
    for relation, count in sorted(report["pair_relations"].items(), key=lambda x: -x[1]):
        print(f"  {count:4d}  {relation:16s} {RELATION_LABELS.get(relation, '')}")
    print(f"\n  минимальный max_diff между РАЗНЫМИ позициями: {report['min_max_diff_between_positions']}")
    print(f"  (порог шума перекодирования — {RECODE_NOISE_MAXDIFF}: всё, что выше, содержит "
          f"хотя бы одну реально различающую деталь)")

    section("САМЫЕ ПОХОЖИЕ ПАРЫ РАЗНЫХ ПОЗИЦИЙ")
    rows = []
    for group in groups:
        if group["type"] == "C_visually_close" and group.get("relation") != "same_file":
            rows.append((group["diff_frac"], group["slugs"], group["relation"], group.get("bands", [])))
        for pair in group.get("pairs", []):
            if pair["relation"] != "same_file":
                rows.append((pair["diff_frac"], pair["slugs"], pair["relation"], pair.get("bands", [])))
    for frac, slugs, relation, bands in sorted(rows)[:15]:
        print(f"  {frac * 100:5.2f}% площади | {relation:14s} | {slugs[0]}")
        print(f"  {'':13s} | {'':14s} | {slugs[1]}")
        if bands:
            band = bands[0]
            print(f"     крупнейшая деталь: y={band['y0_pct']:.2f}..{band['y1_pct']:.2f}, "
                  f"{band['height_px']}x{band['width_px']} px при ширине бутылки {BOTTLE_WIDTH} px")

    section("НАЗВАНИЯ, ПОВТОРЯЮЩИЕСЯ У РАЗНЫХ ВИНОДЕЛЕН")
    n = report["name_collisions"]
    print(f"  названий {n['names']}, позиций {n['slugs']} — текстовый матч без винодельни не работает")

    print(f"\n  групп без единого различающего атрибута: {report['B_same_series']['groups_without_discriminator']}")
    for group in by_type["B_same_series"]:
        if not group["discriminators"]:
            print(f"    {group['key'][:50]:50s} {group['slugs']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="машинный вывод")
    parser.add_argument("--no-visual", action="store_true", help="без сравнения изображений")
    parser.add_argument("--max-visual-pairs", type=int, default=600,
                        help="сколько визуальных кандидатов сравнивать точно (по умолчанию 600)")
    args = parser.parse_args()

    state = build_groups(with_visual=not args.no_visual, max_visual_pairs=args.max_visual_pairs)
    report = build_report(state)
    if args.json:
        payload = {"summary": report,
                   "groups": [{k: v for k, v in g.items() if k != "bank"} for g in state["groups"]]}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print_report(state, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
