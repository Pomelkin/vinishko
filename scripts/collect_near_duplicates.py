#!/usr/bin/env python3
"""Раскладывает near-duplicates Каталога по папкам, чтобы посмотреть их глазами.

Группировку и метрики берёт из scripts/near_duplicates.py, сам занимается только
раскладкой: на каждую группу — своя папка с нормализованными бутылками, коллажем
и кропом той детали, которая пару различает.

    python3 scripts/collect_near_duplicates.py --force
    python3 scripts/collect_near_duplicates.py --out /tmp/nd --width 800
    python3 scripts/collect_near_duplicates.py --no-visual      # только index.md, без картинок

Раскладка:

    data/near_duplicates/
      index.md                                  сводка по всем группам, включая несобранные
      A-one-image-many-slugs/01_<винодельня>__<имя>/
        info.md          поля Каталога, метрики пары, вывод «чем реально отличаются»
        _compare.png     все бутылки группы рядом
        _zoom.png        кроп различающей детали крупно (нет, если различий нет)
        1_<slug>.png     нормализованная бутылка (кроп по альфе, белый фон)
      B-same-series/...
      C-visually-close/...

Требуется Pillow и numpy (кроме режима --no-visual).
"""

from __future__ import annotations

import argparse
import itertools
import re
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import analyze_data as ad  # noqa: E402
import near_duplicates as nd  # noqa: E402

TYPE_DIRS = {
    "A_one_image_many_slugs": ("A-one-image-many-slugs",
                               "A. Один Эталон на несколько позиций — картинкой не разводятся"),
    "B_same_series": ("B-same-series", "B. Одна серия у одного производителя"),
    "C_visually_close": ("C-visually-close", "C. Визуально близкие Эталоны вне одной серии"),
}

CATALOG_FIELDS = ["Название вина", "Категория", "Цвет", "Регион",
                  "Сорт винограда", "Винодельня", "Название фото"]

# Зоны бутылки по доле высоты — чтобы словами сказать, где сидит различающая деталь.
ZONES = [(0.30, "капсула / горлышко"), (0.55, "плечо бутылки"),
         (0.92, "этикетка"), (1.01, "низ этикетки")]

COMPARE_BOX = (1400, 1000)      # коллаж вписывается сюда: иначе бутылки по 1900 px в высоту
ZOOM_BOX = (1400, 1100)         # и зум тоже — 4-кратное увеличение высокой полосы весит мегабайты
ZOOM_TARGET_WIDTH = 620         # ширина одной половины в _zoom.png
ZOOM_PADDING = 40               # поля вокруг различающей детали, px при ширине бутылки 640
ZOOM_MAX_SPAN = 0.35            # в кадр берём и соседние полосы различий, пока они укладываются
                                # в эту долю высоты бутылки — иначе год под этикеткой не попадёт


def folder_slug(text: str, limit: int = 60) -> str:
    """Читаемое имя папки: транслит в латиницу, kebab-case, без хвостовых дефисов."""
    slug = ad.name_to_slug(text).lower().replace("_", "-")
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:limit].strip("-") or "group"


def unique_name(name: str, used: set[str]) -> str:
    candidate, n = name, 2
    while candidate in used:
        candidate = f"{name}-{n}"
        n += 1
    used.add(candidate)
    return candidate


def short_label(group: dict, catalog: dict) -> str:
    """Короткая часть имени папки: серия для B, различающий хвост slug'ов для A и C."""
    if group["type"] == "B_same_series":
        return folder_slug(group["key"].split(" / ", 1)[-1], 46)
    slugs = group["slugs"]
    head = slugs[0]
    if len(slugs) == 2:
        # оставить от второго slug только то, чем он отличается от первого
        prefix = 0
        for a, b in zip(head.split("-"), slugs[1].split("-")):
            if a != b:
                break
            prefix += 1
        tail = "-".join(slugs[1].split("-")[prefix:]) or slugs[1]
        return folder_slug(f"{head[:34]}-vs-{tail[:24]}", 58)
    return folder_slug(head, 52)


def pick_file(group: dict, slug: str, slug_files: dict) -> str | None:
    """Файл-Эталон, которым представлена позиция внутри этой группы."""
    own = slug_files.get(slug) or []
    if not own:
        return None
    for name in group.get("files") or []:               # для A — именно общий файл группы
        if name in own:
            return name
    return own[0]


def group_pairs(group: dict, slug_files: dict) -> list[dict]:
    """Пары внутри группы в едином виде: slugs, files, relation, diff_frac, max_diff, bands."""
    if group["type"] == "B_same_series":
        return list(group.get("pairs") or [])
    if group["type"] == "C_visually_close":
        return [{k: group[k] for k in ("slugs", "files", "relation", "diff_frac",
                                       "max_diff", "bands", "bottle_px") if k in group}]
    pairs = []                                           # A: все позиции сидят на одном файле
    for slug_a, slug_b in itertools.combinations(group["slugs"], 2):
        file_a, file_b = pick_file(group, slug_a, slug_files), pick_file(group, slug_b, slug_files)
        pairs.append({"slugs": [slug_a, slug_b], "files": [file_a, file_b],
                      "relation": "same_file", "diff_frac": 0.0, "max_diff": 0.0, "bands": []})
    return pairs


def zone_of(y0: float, y1: float) -> str:
    middle = (y0 + y1) / 2
    return next(label for bound, label in ZONES if middle < bound)


def describe_difference(pair: dict) -> str:
    """Одной строкой: чем пара реально отличается."""
    relation = pair.get("relation")
    if relation == "same_file":
        return "один и тот же файл на обе позиции — различий нет в принципе"
    if relation == "pixel_identical":
        return "разные файлы, но ни одной различающей детали выше шума перекодирования"
    bands = pair.get("bands") or []
    frac = pair.get("diff_frac", 0.0)
    if not bands:
        return f"различие {frac * 100:.2f}% площади бутылки, сплошной детали не выделяется"
    band = bands[0]
    return (f"различие {frac * 100:.2f}% площади бутылки; крупнейшая деталь "
            f"{band['height_px']}x{band['width_px']} px на высоте "
            f"{band['y0_pct']:.0%}–{band['y1_pct']:.0%} ({zone_of(band['y0_pct'], band['y1_pct'])}), "
            f"при ширине бутылки {nd.BOTTLE_WIDTH} px")


def best_pair(pairs: list[dict]) -> dict | None:
    """Самая похожая пара с двумя разными файлами — её и показываем крупно."""
    usable = [p for p in pairs
              if p.get("relation") not in (None, "same_file") and all(p.get("files") or [None])]
    return min(usable, key=lambda p: p.get("diff_frac", 1.0)) if usable else None


# --- картинки ---------------------------------------------------------------

def render_bottles(bank, files: list[str], width: int) -> list:
    return [bank.normalized(name, width) for name in files]


def compare_sheet(bank, images: list, gap: int = 12):
    """Все бутылки группы рядом на белом фоне, вписанные в COMPARE_BOX."""
    Image = bank._Image
    if not images:
        return None
    width = max(im.width for im in images)
    height = max(im.height for im in images)
    sheet = Image.new("RGB", (width * len(images) + gap * (len(images) - 1), height), (255, 255, 255))
    for i, im in enumerate(images):
        sheet.paste(im, (i * (width + gap), 0))
    sheet.thumbnail(COMPARE_BOX, Image.LANCZOS)
    return sheet


def zoom_window(bands: list[tuple], height: int) -> tuple[int, int, int, int]:
    """Окно кропа: крупнейшая полоса различий плюс соседние, если те рядом.

    Отличие часто распадается на несколько полос — у Primum Alveus это римская цифра
    на горлышке и год под этикеткой. Показать только крупнейшую значит потерять год.
    """
    ordered = sorted(bands, key=lambda b: -b[4])
    y0, y1, x0, x1, _ = ordered[0]
    for band in ordered[1:]:
        new_y0, new_y1 = min(y0, band[0]), max(y1, band[1])
        if (new_y1 - new_y0) / height > ZOOM_MAX_SPAN:
            continue
        y0, y1 = new_y0, new_y1
        x0, x1 = min(x0, band[2]), max(x1, band[3])
    return y0, y1, x0, x1


def zoom_sheet(bank, pair: dict, width: int):
    """Кроп различающих деталей из обеих бутылок, бок о бок и с увеличением."""
    Image = bank._Image
    file_a, file_b = pair["files"]
    diff_map, _ = nd.aligned_difference(bank, file_a, file_b)
    bands = nd.diff_bands(diff_map)
    if not bands:
        return None
    height, bottle_width = diff_map.shape
    y0, y1, x0, x1 = zoom_window(bands, height)

    # координаты посчитаны на выровненной бутылке шириной nd.BOTTLE_WIDTH — переводим в доли
    pad = ZOOM_PADDING / nd.BOTTLE_WIDTH
    box = (max(0.0, x0 / bottle_width - pad), max(0.0, y0 / height - pad),
           min(1.0, (x1 + 1) / bottle_width + pad), min(1.0, (y1 + 1) / height + pad))

    crops = []
    for name in (file_a, file_b):
        im = bank.normalized(name, width)
        usable_height = min(im.height, round(width * height / bottle_width))
        crop = im.crop((round(box[0] * im.width), round(box[1] * usable_height),
                        max(1, round(box[2] * im.width)), max(2, round(box[3] * usable_height))))
        scale = min(4.0, ZOOM_TARGET_WIDTH / max(1, crop.width))
        crops.append(crop.resize((max(1, round(crop.width * scale)),
                                 max(1, round(crop.height * scale))), Image.LANCZOS))
    gap, canvas_h = 16, max(c.height for c in crops)
    sheet = Image.new("RGB", (sum(c.width for c in crops) + gap, canvas_h), (255, 255, 255))
    offset = 0
    for crop in crops:
        sheet.paste(crop, (offset, 0))
        offset += crop.width + gap
    sheet.thumbnail(ZOOM_BOX, Image.LANCZOS)
    return sheet


# --- info.md ----------------------------------------------------------------

def escape_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def write_info(path: Path, group: dict, catalog: dict, slug_files: dict,
               pairs: list[dict], images: dict[str, str], shown_pair: dict | None) -> None:
    _, type_title = TYPE_DIRS[group["type"]]
    lines = [f"# {escape_cell(group['key'])}", "",
             f"**Тип группы:** {type_title}", "",
             f"**Винодельня:** {', '.join(group['wineries'])}", "",
             f"**Позиций в группе:** {len(group['slugs'])}, "
             f"из них с Эталоном: {sum(1 for s in group['slugs'] if slug_files.get(s))}", "",
             f"**Различающие атрибуты:** "
             f"{', '.join(group['discriminators']) or 'НЕТ — ни один атрибут не разводит позиции'}", ""]

    if pairs:
        headline = describe_difference(min(pairs, key=lambda p: p.get("diff_frac", 1.0)))
        lines += [f"**Чем реально отличаются:** {headline}", ""]

    lines += ["## Позиции", ""]
    for i, slug in enumerate(group["slugs"], 1):
        row = catalog[slug]
        lines += [f"### {i}. `{slug}`", "", "| поле | значение |", "|---|---|"]
        for field in CATALOG_FIELDS:
            lines.append(f"| {field} | {escape_cell(row[field])} |")
        picture = images.get(slug)
        lines.append(f"| Эталон | {'`' + picture + '`' if picture else '**нет в Медиа-дампе**'} |")
        lines.append("")

    if pairs:
        lines += ["## Метрики пар", "",
                  "| пара | relation | различие, % площади | max_diff | крупнейшая деталь |",
                  "|---|---|---|---|---|"]
        for pair in sorted(pairs, key=lambda p: p.get("diff_frac", 1.0)):
            bands = pair.get("bands") or []
            detail = "—"
            if bands:
                band = bands[0]
                detail = (f"{band['height_px']}x{band['width_px']} px, y="
                          f"{band['y0_pct']:.0%}–{band['y1_pct']:.0%} ({zone_of(band['y0_pct'], band['y1_pct'])})")
            lines.append(f"| {pair['slugs'][0]} ↔ {pair['slugs'][1]} | {pair.get('relation', '—')} "
                         f"| {pair.get('diff_frac', 0.0) * 100:.2f} | {pair.get('max_diff', 0.0)} "
                         f"| {detail} |")
        lines.append("")
        lines += [f"Пороги: `same_file` — один файл; `pixel_identical` — max_diff < "
                  f"{nd.RECODE_NOISE_MAXDIFF}; `micro_diff` — различие < {nd.MICRO_DIFF_FRAC:.1%} "
                  f"площади; `small_diff` — < {nd.SMALL_DIFF_FRAC:.0%}.", ""]

    if shown_pair:
        lines += [f"`_zoom.png` показывает пару "
                  f"`{shown_pair['slugs'][0]}` ↔ `{shown_pair['slugs'][1]}` — "
                  f"самую похожую в группе.", ""]
        bands = shown_pair.get("bands") or []
        if bands:
            lines += ["Все различающие полосы этой пары (сверху вниз):", ""]
            for band in sorted(bands, key=lambda b: b["y0_pct"]):
                lines.append(f"- y={band['y0_pct']:.0%}–{band['y1_pct']:.0%} "
                             f"({zone_of(band['y0_pct'], band['y1_pct'])}): "
                             f"{band['height_px']}x{band['width_px']} px, {band['area_px']} px различий")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# --- сборка -----------------------------------------------------------------

def collect(out_dir: Path, width: int, with_visual: bool, max_visual_pairs: int,
            max_visual_groups: int) -> dict:
    state = nd.build_groups(with_visual=with_visual, max_visual_pairs=max_visual_pairs)
    catalog, slug_files, bank = state["catalog"], state["slug_files"], state["bank"]

    by_type: dict[str, list[dict]] = {key: [] for key in TYPE_DIRS}
    for group in state["groups"]:
        by_type[group["type"]].append(group)

    # порядок папок: самое похожее — первым, чтобы интересное было в начале списка
    by_type["A_one_image_many_slugs"].sort(key=lambda g: g["slugs"])
    by_type["C_visually_close"].sort(key=lambda g: (g.get("diff_frac", 1.0), g["slugs"]))

    def series_rank(group):
        pairs = group.get("pairs") or []
        return (min((p["diff_frac"] for p in pairs), default=1.0), group["key"])

    by_type["B_same_series"].sort(key=series_rank)

    imageless = [g for g in by_type["B_same_series"]
                 if not any(slug_files.get(s) for s in g["slugs"])]
    collected: dict[str, list[tuple[str, dict, list[dict]]]] = {key: [] for key in TYPE_DIRS}
    skipped_visual: list[dict] = []
    stats = {"folders": 0, "images": 0, "compare": 0, "zoom": 0}

    for type_key, groups in by_type.items():
        dir_name = TYPE_DIRS[type_key][0]
        used_names: set[str] = set()
        rank = 0
        for group in groups:
            files_by_slug = {s: pick_file(group, s, slug_files) for s in group["slugs"]}
            if not any(files_by_slug.values()):
                continue                                   # смотреть нечего — только в index.md
            rank += 1
            if type_key == "C_visually_close" and max_visual_groups and rank > max_visual_groups:
                skipped_visual.append(group)
                continue

            winery = folder_slug(group["wineries"][0], 24)
            name = unique_name(f"{rank:03d}_{winery}__{short_label(group, catalog)}", used_names)
            folder = out_dir / dir_name / name
            folder.mkdir(parents=True, exist_ok=True)

            pairs = group_pairs(group, slug_files)
            written: dict[str, str] = {}
            images = []
            if bank is not None:
                for i, slug in enumerate(group["slugs"], 1):
                    source = files_by_slug[slug]
                    if not source:
                        continue
                    picture = bank.normalized(source, width)
                    picture.save(folder / f"{i}_{slug[:70]}.png")
                    written[slug] = source
                    images.append(picture)
                    stats["images"] += 1
                sheet = compare_sheet(bank, images)
                if sheet is not None:
                    sheet.save(folder / "_compare.png")
                    stats["compare"] += 1

            shown = best_pair(pairs) if bank is not None else None
            if shown is not None:
                zoom = zoom_sheet(bank, shown, width)
                if zoom is not None:
                    zoom.save(folder / "_zoom.png")
                    stats["zoom"] += 1
                else:
                    shown = None

            write_info(folder / "info.md", group, catalog, slug_files, pairs, written, shown)
            collected[type_key].append((f"{dir_name}/{name}", group, pairs))
            stats["folders"] += 1

    write_index(out_dir, collected, imageless, skipped_visual, catalog, slug_files, stats,
                width, with_visual, max_visual_groups)
    return stats


def write_index(out_dir: Path, collected, imageless, skipped_visual, catalog, slug_files,
                stats, width, with_visual, max_visual_groups) -> None:
    lines = ["# Near-duplicates Каталога — галерея", "",
             "Собрано `scripts/collect_near_duplicates.py`, группировка и метрики — "
             "`scripts/near_duplicates.py`. Разбор — `docs/NEAR_DUPLICATES.md`.", "",
             f"Папок: {stats['folders']}, картинок-бутылок: {stats['images']}, "
             f"коллажей: {stats['compare']}, зумов различий: {stats['zoom']}. "
             f"Ширина бутылки: {width} px." + ("" if with_visual else " Режим без картинок."), "",
             "В каждой папке: `_compare.png` — бутылки рядом, `_zoom.png` — та деталь, "
             "которая пару различает, `info.md` — поля Каталога и метрики.", ""]

    for type_key, (dir_name, title) in TYPE_DIRS.items():
        rows = collected[type_key]
        lines += [f"## {title}", "", f"Собрано групп: {len(rows)}", "",
                  "| папка | винодельня | позиции | relation | различие, % площади | различают атрибуты |",
                  "|---|---|---|---|---|---|"]
        for path, group, pairs in rows:
            pair = min(pairs, key=lambda p: p.get("diff_frac", 1.0)) if pairs else {}
            lines.append(
                f"| [{path.split('/')[-1]}]({path}/info.md) "
                f"| {escape_cell(', '.join(group['wineries']))} "
                f"| {' <br> '.join('`' + s + '`' for s in group['slugs'])} "
                f"| {pair.get('relation', '—')} "
                f"| {pair.get('diff_frac', 0.0) * 100:.2f} "
                f"| {escape_cell(', '.join(group['discriminators']) or '—')} |")
        lines.append("")

    if skipped_visual:
        lines += [f"## Визуально близкие пары без папок (лимит `--max-visual-groups "
                  f"{max_visual_groups}`)", "",
                  f"Ещё {len(skipped_visual)} пар прошли порог визуальной близости, но различаются "
                  "заметнее собранных — папки для них не создавались. Снять лимит: "
                  "`--max-visual-groups 0`.", "",
                  "| позиции | винодельня | relation | различие, % площади |", "|---|---|---|---|"]
        for group in skipped_visual:
            lines.append(f"| {' ↔ '.join('`' + s + '`' for s in group['slugs'])} "
                         f"| {escape_cell(', '.join(group['wineries']))} "
                         f"| {group.get('relation', '—')} "
                         f"| {group.get('diff_frac', 0.0) * 100:.2f} |")
        lines.append("")

    if imageless:
        lines += ["## Группы без единого Эталона — смотреть нечего", "",
                  f"{len(imageless)} серий, в которых ни у одной позиции нет изображения "
                  "в Медиа-дампе. Развести их можно только текстом.", "",
                  "| серия | позиции | различают атрибуты |", "|---|---|---|"]
        for group in imageless:
            lines.append(f"| {escape_cell(group['key'])} "
                         f"| {' <br> '.join('`' + s + '`' for s in group['slugs'])} "
                         f"| {escape_cell(', '.join(group['discriminators']) or '—')} |")
        lines.append("")

    (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO / "data" / "near_duplicates",
                        help="куда складывать (по умолчанию data/near_duplicates/ в репозитории)")
    parser.add_argument("--force", action="store_true", help="перезаписать непустую папку")
    parser.add_argument("--width", type=int, default=600, help="ширина бутылки в px (по умолчанию 600)")
    parser.add_argument("--no-visual", action="store_true",
                        help="без картинок и сравнения — только index.md по текстовым группам")
    parser.add_argument("--max-visual-pairs", type=int, default=600,
                        help="сколько визуальных кандидатов сравнивать точно")
    parser.add_argument("--max-visual-groups", type=int, default=150,
                        help="сколько групп типа C раскладывать по папкам, 0 — все (по умолчанию 150)")
    args = parser.parse_args()

    out_dir: Path = args.out
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.force:
            print(f"{out_dir} уже существует и не пуста — запустите с --force, чтобы перезаписать",
                  file=sys.stderr)
            return 1
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    stats = collect(out_dir, args.width, not args.no_visual,
                    args.max_visual_pairs, args.max_visual_groups)
    total = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    print(f"готово за {time.time() - started:.0f} c: папок {stats['folders']}, "
          f"бутылок {stats['images']}, коллажей {stats['compare']}, зумов {stats['zoom']}; "
          f"{total / 1024 / 1024:.0f} МБ в {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
