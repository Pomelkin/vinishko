#!/usr/bin/env python3
"""Переименовывает файлы в data/strapi/img/ в исходные имена из колонки
«Название фото» дампа Strapi — in-place.

Правило ровно одно, обратное к тому, как Strapi формирует `hash` при загрузке:

    <файл в дампе> = nameToSlug(basename(«Название фото»)) + "_" + <10 hex> + <ext>

где nameToSlug — `@sindresorhus/slugify` с separator="_", lowercase=false
(транслитерация → decamelize → замена всего не-[A-Za-z0-9] на "_").

Скрипт строит slug для каждого «Название фото» из CSV, снимает с имени файла
десятисимвольный hex-суффикс и сопоставляет строки точно. Ничего похожего,
никакого fuzzy: файл, для которого нет точного совпадения, не трогается —
части эталонов в дампе просто нет.

По умолчанию — сухой прогон. Реальное переименование включается флагом --apply;
он же пишет журнал (--journal), по которому переименование откатывается --revert.

    python3 scripts/rename_img_to_csv_names.py                 # что будет сделано
    python3 scripts/rename_img_to_csv_names.py --apply         # переименовать
    python3 scripts/rename_img_to_csv_names.py --revert scripts/rename_journal.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "data" / "strapi" / "strapi_output0709.csv"
DEFAULT_IMG_DIR = REPO_ROOT / "data" / "strapi" / "img"
DEFAULT_JOURNAL = REPO_ROOT / "scripts" / "rename_journal.json"

PHOTO_COLUMN = "Название фото"

# Суффикс, который Strapi добавляет к слагу: crypto.randomBytes(5).toString('hex').
HASH_SUFFIX_RE = re.compile(r"_[0-9a-f]{10}$")

# Транслитерация кириллицы как в @sindresorhus/transliterate.
# Таблица подтверждена на данных дампа: «Белое брют» → Beloe_bryut,
# «ДНК Аура Меркотан2» → DNK_Aura_Merkotan2, «Цимлянское…» → Czimlyanskoe…
_TRANSLIT_BASE = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "j", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "cz", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}
TRANSLIT = {}
for _lower, _latin in _TRANSLIT_BASE.items():
    TRANSLIT[_lower] = _latin
    TRANSLIT[_lower.upper()] = _latin.capitalize() if _latin else ""


def transliterate(text: str) -> str:
    return "".join(TRANSLIT.get(ch, ch) for ch in text)


def decamelize(text: str) -> str:
    """Порт decamelize из @sindresorhus/slugify: вставляет пробелы на границах регистра."""
    text = re.sub(r"([A-Z]{2,})(\d+)", r"\1 \2", text)
    text = re.sub(r"([a-z\d]+)([A-Z]{2,})", r"\1 \2", text)
    text = re.sub(r"([a-z\d])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z\d]+)", r"\1 \2", text)
    return text


def name_to_slug(name: str) -> str:
    """Strapi nameToSlug(name, {separator: '_', lowercase: false})."""
    slug = decamelize(transliterate(unicodedata.normalize("NFC", name)))
    slug = re.sub(r"[^a-zA-Z\d]+", "_", slug)
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("_")


def strip_hash(stem: str) -> str:
    return HASH_SUFFIX_RE.sub("", stem)


def sanitize(name: str) -> str:
    """Имя файла не должно содержать разделитель пути и NUL."""
    return unicodedata.normalize("NFC", name).replace("/", "_").replace("\0", "")


def load_catalog(csv_path: Path) -> dict[str, str]:
    """Ключ сопоставления → «Название фото» (именно оно станет именем файла)."""
    catalog: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if PHOTO_COLUMN not in (reader.fieldnames or []):
            sys.exit(f"В {csv_path} нет колонки «{PHOTO_COLUMN}»")
        for row in reader:
            photo = (row[PHOTO_COLUMN] or "").strip()
            if not photo:
                continue
            key = name_to_slug(os.path.splitext(photo)[0])
            if key:
                catalog.setdefault(key, photo)
    return catalog


def plan_renames(img_dir: Path, catalog: dict[str, str]) -> tuple[list[dict], list[str]]:
    """Возвращает (план переименований, список файлов без совпадения)."""
    files = sorted(p.name for p in img_dir.iterdir() if p.is_file() and not p.name.startswith("."))

    by_key: dict[str, list[str]] = defaultdict(list)
    unmatched: list[str] = []
    for filename in files:
        stem, _ = os.path.splitext(filename)
        key = strip_hash(stem)
        if key in catalog:
            by_key[key].append(filename)
        else:
            unmatched.append(filename)

    existing = set(files)
    taken: set[str] = set()
    plan: list[dict] = []
    for key in sorted(by_key):
        photo_name = catalog[key]
        target_stem, target_ext = os.path.splitext(sanitize(photo_name))
        for index, filename in enumerate(by_key[key]):
            # Расширение берём фактическое: в дампе встречаются .jpg/.png там,
            # где CSV обещает .webp. Совпадает в подавляющем большинстве случаев.
            src_ext = os.path.splitext(filename)[1] or target_ext
            candidate = f"{target_stem}{src_ext}"
            # Один ключ может быть у нескольких загрузок (разный hex) — второй
            # и далее получают детерминированный суффикс, чтобы ничего не потерять.
            bump = 2
            while candidate in taken or (candidate in existing and candidate != filename):
                candidate = f"{target_stem} ({bump}){src_ext}"
                bump += 1
            taken.add(candidate)
            plan.append(
                {
                    "src": filename,
                    "dst": candidate,
                    "csv_name": photo_name,
                    "duplicate_index": index,
                }
            )
    return plan, unmatched


def apply_renames(img_dir: Path, plan: list[dict]) -> list[dict]:
    """Переименовывает in-place. Двухфазно, через временные имена, чтобы
    выдержать циклы и перестановки src/dst внутри одного каталога."""
    done: list[dict] = []
    staged: list[tuple[Path, dict]] = []
    for item in plan:
        if item["src"] == item["dst"]:
            continue
        src = img_dir / item["src"]
        tmp = img_dir / f".rename_tmp_{os.getpid()}_{len(staged)}"
        src.rename(tmp)
        staged.append((tmp, item))
    for tmp, item in staged:
        dst = img_dir / item["dst"]
        if dst.exists():
            tmp.rename(img_dir / item["src"])  # откат этого шага
            print(f"! занято, пропуск: {item['src']} -> {item['dst']}", file=sys.stderr)
            continue
        tmp.rename(dst)
        done.append({"src": item["src"], "dst": item["dst"]})
    return done


def revert(img_dir: Path, journal_path: Path) -> int:
    records = json.loads(journal_path.read_text(encoding="utf-8"))["renames"]
    reverted = 0
    for record in reversed(records):
        current = img_dir / record["dst"]
        original = img_dir / record["src"]
        if not current.exists():
            print(f"! нет файла {record['dst']}, пропуск", file=sys.stderr)
            continue
        if original.exists():
            print(f"! {record['src']} уже занят, пропуск", file=sys.stderr)
            continue
        current.rename(original)
        reverted += 1
    return reverted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="дамп Strapi (CSV)")
    parser.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR, help="каталог с файлами дампа")
    parser.add_argument("--apply", action="store_true", help="выполнить переименование (по умолчанию сухой прогон)")
    parser.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL, help="куда писать журнал переименований")
    parser.add_argument("--revert", type=Path, metavar="JOURNAL", help="откатить переименование по журналу")
    parser.add_argument("--limit", type=int, default=20, help="сколько строк плана показать (0 — все)")
    args = parser.parse_args()

    if not args.img_dir.is_dir():
        sys.exit(f"Каталог не найден: {args.img_dir}")

    if args.revert:
        count = revert(args.img_dir, args.revert)
        print(f"Откачено файлов: {count}")
        return 0

    catalog = load_catalog(args.csv)
    plan, unmatched = plan_renames(args.img_dir, catalog)
    total_files = len(plan) + len(unmatched)

    shown = plan if args.limit == 0 else plan[: args.limit]
    for item in shown:
        mark = " [дубль]" if item["duplicate_index"] else ""
        print(f"{item['src']}  ->  {item['dst']}{mark}")
    if len(shown) < len(plan):
        print(f"... и ещё {len(plan) - len(shown)} (--limit 0, чтобы показать все)")

    print()
    print(f"CSV: уникальных «Название фото» — {len(catalog)}")
    print(f"Файлов в каталоге: {total_files}")
    print(f"Сопоставлено: {len(plan)}")
    print(f"Без совпадения (останутся как есть): {len(unmatched)}")

    if not args.apply:
        print("\nСухой прогон. Запустите с --apply, чтобы переименовать.")
        return 0

    done = apply_renames(args.img_dir, plan)
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    args.journal.write_text(
        json.dumps({"img_dir": str(args.img_dir), "renames": done}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nПереименовано: {len(done)}")
    print(f"Журнал: {args.journal} (откат: --revert {args.journal})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
