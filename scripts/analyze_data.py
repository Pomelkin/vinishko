#!/usr/bin/env python3
"""Считает статистику по данным кейса: каталог, медиа-дамп, их связывание, eval-пакет.

Все цифры в docs/DATA_RESEARCH.md получены этим скриптом — он же служит
регрессией: если дамп заменят, расхождение будет видно сразу.

    python3 scripts/analyze_data.py            # текстовый отчёт
    python3 scripts/analyze_data.py --json     # машинный формат

Требуется Pillow (только для секции изображений; без него она пропускается).
Учитывает, что scripts/rename_img_to_csv_names.py мог переименовать часть
файлов: исходные strapi-имена восстанавливаются по журналу.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import re
import statistics
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "data" / "strapi" / "strapi_output0709.csv"
IMG_DIR = REPO / "data" / "strapi" / "img"
JOURNAL = REPO / "scripts" / "rename_journal.json"
EVAL_DIR = REPO / "data" / "eval"

HEX_SUFFIX = re.compile(r"_[0-9a-f]{10}$")

# Транслитерация @sindresorhus/transliterate, которую использует Strapi в nameToSlug.
_TR_BASE = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "j", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "cz", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}
TRANSLIT = {}
for _low, _lat in _TR_BASE.items():
    TRANSLIT[_low] = _lat
    TRANSLIT[_low.upper()] = _lat.capitalize() if _lat else ""


def decamelize(text: str) -> str:
    text = re.sub(r"([A-Z]{2,})(\d+)", r"\1 \2", text)
    text = re.sub(r"([a-z\d]+)([A-Z]{2,})", r"\1 \2", text)
    text = re.sub(r"([a-z\d])([A-Z])", r"\1 \2", text)
    return re.sub(r"([A-Z]+)([A-Z][a-z\d]+)", r"\1 \2", text)


def name_to_slug(name: str) -> str:
    """Strapi nameToSlug(name, {separator: '_', lowercase: false})."""
    s = "".join(TRANSLIT.get(c, c) for c in unicodedata.normalize("NFC", name))
    s = re.sub(r"[^a-zA-Z\d]+", "_", decamelize(s))
    return re.sub(r"_+", "_", s).strip("_")


def loose_key(name: str) -> str:
    """Агрессивная нормализация для проверки «а вдруг правило Strapi не то»."""
    s = unicodedata.normalize("NFC", name).lower()
    s = "".join(_TR_BASE.get(c, c) for c in s)
    return re.sub(r"[^a-z0-9]+", "", s)


def load_catalog() -> dict[str, dict]:
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    catalog: dict[str, dict] = {}
    for row in rows:
        catalog.setdefault(row["Slug"].strip(), row)
    load_catalog.raw_rows = len(rows)  # type: ignore[attr-defined]
    return catalog


def original_names() -> dict[str, str]:
    """Текущее имя файла -> исходное имя из strapi-дампа."""
    dst2src = {}
    if JOURNAL.exists():
        dst2src = {
            r["dst"]: r["src"]
            for r in json.loads(JOURNAL.read_text(encoding="utf-8"))["renames"]
        }
    return {f: dst2src.get(f, f) for f in os.listdir(IMG_DIR) if not f.startswith(".")}


def link_catalog_to_images(catalog, orig) -> dict:
    by_file_key = collections.defaultdict(list)
    for current, source in orig.items():
        stem = os.path.splitext(source)[0]
        by_file_key[HEX_SUFFIX.sub("", stem)].append(current)

    matched, unmatched = {}, []
    for slug, row in catalog.items():
        key = name_to_slug(os.path.splitext(row["Название фото"].strip())[0])
        if key in by_file_key:
            matched[slug] = by_file_key[key]
        else:
            unmatched.append(slug)

    # Второй канал: имя файла само похоже на slug позиции.
    slug_key = {re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-"): s for s in catalog}
    by_slug = {}
    for current, source in orig.items():
        key = re.sub(r"[^a-z0-9]+", "-", HEX_SUFFIX.sub("", os.path.splitext(source)[0]).lower()).strip("-")
        if key in slug_key:
            by_slug.setdefault(slug_key[key], []).append(current)

    # Третий канал: агрессивная нормализация по остатку — проверка на «правило не то».
    loose_files = collections.defaultdict(list)
    for current, source in orig.items():
        loose_files[loose_key(HEX_SUFFIX.sub("", os.path.splitext(source)[0]))].append(current)
    loose_hits = sum(
        1 for slug in unmatched
        if loose_key(os.path.splitext(catalog[slug]["Название фото"])[0]) in loose_files
    )

    covered = set(matched) | set(by_slug)
    return {
        "by_photo_name": matched,
        "by_slug_name": by_slug,
        "covered": covered,
        "uncovered": sorted(set(catalog) - covered),
        "loose_extra_hits": loose_hits,
        "files_total": len(orig),
        "files_used": len({f for fs in matched.values() for f in fs} | {f for fs in by_slug.values() for f in fs}),
    }


def image_stats(names: list[str]) -> dict:
    try:
        from PIL import Image
    except ImportError:
        return {}
    Image.MAX_IMAGE_PIXELS = None
    w, h, ar, size, bottle_w = [], [], [], [], []
    modes, fmts, unreadable = collections.Counter(), collections.Counter(), 0
    for name in names:
        path = IMG_DIR / name
        size.append(path.stat().st_size)
        try:
            with Image.open(path) as im:
                w.append(im.width)
                h.append(im.height)
                ar.append(im.width / im.height)
                modes[im.mode] += 1
                fmts[im.format] += 1
                if im.mode == "RGBA":
                    box = im.getchannel("A").getbbox()
                    bottle_w.append(box[2] - box[0] if box else im.width)
                else:
                    bottle_w.append(im.width)
        except Exception:
            unreadable += 1
    if not w:
        return {}
    pct = lambda a, p: sorted(a)[int(p * (len(a) - 1))]
    return {
        "n": len(names), "unreadable": unreadable,
        "formats": dict(fmts), "modes": dict(modes),
        "width_p10_p50_p90": [pct(w, .1), pct(w, .5), pct(w, .9)],
        "height_p10_p50_p90": [pct(h, .1), pct(h, .5), pct(h, .9)],
        "aspect_p10_p50_p90": [round(pct(ar, .1), 2), round(pct(ar, .5), 2), round(pct(ar, .9), 2)],
        "kb_p50_p90": [pct(size, .5) // 1024, pct(size, .9) // 1024],
        "bottle_width_p10_p50_p90": [pct(bottle_w, .1), pct(bottle_w, .5), pct(bottle_w, .9)],
        "bottle_narrower_than_400px": sum(1 for x in bottle_w if x < 400),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="машинный вывод")
    args = parser.parse_args()

    catalog = load_catalog()
    raw_rows = load_catalog.raw_rows  # type: ignore[attr-defined]
    orig = original_names()
    link = link_catalog_to_images(catalog, orig)

    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        slug_multiplicity = collections.Counter(row["Slug"].strip() for row in csv.DictReader(fh))

    ref_files = sorted({f for fs in link["by_photo_name"].values() for f in fs}
                       | {f for fs in link["by_slug_name"].values() for f in fs})
    free_files = sorted(set(orig) - set(ref_files))

    # один файл — несколько позиций: коллизии привязки либо алиасы одного объекта
    digest = {f: hashlib.sha256((IMG_DIR / f).read_bytes()).hexdigest() for f in ref_files}
    slug_files = collections.defaultdict(set)
    for src in (link["by_photo_name"], link["by_slug_name"]):
        for slug, files in src.items():
            slug_files[slug].update(files)
    by_digest = collections.defaultdict(set)
    for slug, files in slug_files.items():
        for f in files:
            by_digest[digest[f]].add(slug)
    shared = {k: sorted(v) for k, v in by_digest.items() if len(v) > 1}

    # Object-level audit uses one authoritative reference: explicit photo name first,
    # filename-to-slug only as a fallback.  The union above is retained to account for
    # every file, but it creates two avoidable collision groups.
    preferred_slug_files = {
        slug: (link["by_photo_name"].get(slug) or link["by_slug_name"].get(slug) or [])
        for slug in catalog
    }
    preferred_by_digest = collections.defaultdict(set)
    for slug, files in preferred_slug_files.items():
        for filename in files[:1]:
            preferred_by_digest[digest[filename]].add(slug)
    preferred_shared = {
        key: sorted(slugs) for key, slugs in preferred_by_digest.items() if len(slugs) > 1
    }

    coverage_by_winery = collections.defaultdict(lambda: [0, 0])
    coverage_by_region = collections.defaultdict(lambda: [0, 0])
    for slug, row in catalog.items():
        hit = 0 if slug in link["covered"] else 1
        coverage_by_winery[row["Винодельня"].strip()][hit] += 1
        coverage_by_region[row["Регион"].strip()][hit] += 1

    report = {
        "catalog": {
            "csv_rows": raw_rows,
            "unique_slugs": len(catalog),
            "slugs_duplicated_twice": sum(1 for v in slug_multiplicity.values() if v == 2),
            "slugs_single_row": sum(1 for v in slug_multiplicity.values() if v == 1),
            "unique_wine_names": len({r["Название вина"].strip() for r in catalog.values()}),
            "unique_wineries": len({r["Винодельня"].strip() for r in catalog.values()}),
            "unique_regions": len({r["Регион"].strip() for r in catalog.values()}),
            "unique_photo_names": len({r["Название фото"].strip() for r in catalog.values()}),
            "categories": dict(collections.Counter(r["Категория"].strip() for r in catalog.values())),
            "description_len_p50": int(statistics.median(len(r["Описание"]) for r in catalog.values())),
            "names_with_stray_spaces": sum(1 for r in catalog.values()
                                           if r["Название вина"] != r["Название вина"].strip()),
            "multiline_descriptions": sum(1 for r in catalog.values() if "\n" in r["Описание"]),
        },
        "media": {
            "files_total": link["files_total"],
            "extensions": dict(collections.Counter(
                os.path.splitext(v)[1].lower() for v in orig.values()).most_common()),
        },
        "linking": {
            "covered_slugs": len(link["covered"]),
            "uncovered_slugs": len(link["uncovered"]),
            "coverage_pct": round(100 * len(link["covered"]) / len(catalog), 1),
            "matched_by_photo_name": len(link["by_photo_name"]),
            "matched_by_slug_name": len(link["by_slug_name"]),
            "extra_hits_from_loose_matching": link["loose_extra_hits"],
            "reference_files": len(ref_files),
            "unrelated_files": len(free_files),
            "shared_digest_groups_union_linking": len(shared),
            "slugs_in_shared_groups_union_linking": len({s for v in shared.values() for s in v}),
            "shared_digest_groups_preferred_linking": len(preferred_shared),
            "slugs_in_shared_groups_preferred_linking": len({
                slug for slugs in preferred_shared.values() for slug in slugs
            }),
            "wineries_fully_uncovered": sum(1 for a, b in coverage_by_winery.values() if a == 0),
            "wineries_fully_covered": sum(1 for a, b in coverage_by_winery.values() if b == 0),
        },
        "coverage_by_region": {k: {"with_photo": v[0], "without_photo": v[1]}
                               for k, v in sorted(coverage_by_region.items(), key=lambda x: -sum(x[1]))},
        "top_gaps_by_winery": [{"winery": w, "without_photo": v[1], "with_photo": v[0]}
                               for w, v in sorted(coverage_by_winery.items(), key=lambda x: -x[1][1])[:20]],
        "reference_images": image_stats(ref_files),
        "unrelated_images": image_stats([f for f in free_files
                                         if os.path.splitext(f)[1].lower() in {".webp", ".jpg", ".jpeg", ".png"}]),
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    def section(title):
        print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")

    section("КАТАЛОГ")
    for k, v in report["catalog"].items():
        print(f"  {k:32s} {v}")
    section("МЕДИА-ДАМП")
    for k, v in report["media"].items():
        print(f"  {k:32s} {v}")
    section("СВЯЗЫВАНИЕ slug ↔ изображение")
    for k, v in report["linking"].items():
        print(f"  {k:32s} {v}")
    section("ПОКРЫТИЕ ПО РЕГИОНАМ")
    for region, v in report["coverage_by_region"].items():
        total = v["with_photo"] + v["without_photo"]
        print(f"  {region[:28]:28s} {v['with_photo']:5d} / {total:5d}  ({100 * v['with_photo'] / total:.0f}%)")
    section("ТОП ПРОБЕЛОВ ПО ВИНОДЕЛЬНЯМ")
    for item in report["top_gaps_by_winery"]:
        print(f"  без фото {item['without_photo']:4d} | есть {item['with_photo']:4d} | {item['winery']}")
    if report["reference_images"]:
        section("ИЗОБРАЖЕНИЯ")
        print("  эталоны:", json.dumps(report["reference_images"], ensure_ascii=False))
        print("  прочие: ", json.dumps(report["unrelated_images"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
