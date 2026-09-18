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

По умолчанию — сухой прогон. Скрипт идемпотентен: при повторном запуске он читает
существующий журнал, узнаёт уже переименованные файлы и не добавляет новые суффиксы.
Реальное переименование включается флагом --apply; журнал накапливает исходное имя
каждого файла и не теряется при повторном запуске. Неоднозначные ключи пропускаются,
а не исправляются через ``setdefault``.

    python3 scripts/rename_img_to_csv_names.py                 # что будет сделано
    python3 scripts/rename_img_to_csv_names.py --check         # проверить идемпотентность
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


def load_catalog(csv_path: Path) -> dict[str, list[str]]:
    """Ключ сопоставления → все уникальные значения «Название фото».

    Обычно значений одно. Если будущий дамп даст одному Strapi-ключу разные
    имена, неоднозначность должна быть видна вызывающему коду: молча брать
    первую строку здесь нельзя.
    """
    grouped: dict[str, set[str]] = defaultdict(set)
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
                grouped[key].add(photo)
    return {key: sorted(values) for key, values in grouped.items()}


def journal_sources(journal_path: Path) -> dict[str, str]:
    """Текущее имя → самое первое имя файла в сыром Strapi-дампе."""
    if not journal_path.exists():
        return {}
    payload = json.loads(journal_path.read_text(encoding="utf-8"))
    return {record["dst"]: record["src"] for record in payload.get("renames", [])}


def remove_copy_suffix(stem: str) -> str:
    """Remove only the deterministic `` (N)`` suffix made by this script."""
    return re.sub(r" \(\d+\)$", "", stem)


def plan_renames(
    img_dir: Path,
    catalog: dict[str, list[str]],
    journal_path: Path = DEFAULT_JOURNAL,
) -> tuple[list[dict], list[str], list[dict]]:
    """Return a safe, idempotent plan, unmatched files and ambiguities."""
    files = sorted(p.name for p in img_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    origins = journal_sources(journal_path)

    by_target: dict[str, list[str]] = defaultdict(list)
    unmatched: list[str] = []
    ambiguous: list[dict] = []
    for filename in files:
        source_stem = strip_hash(os.path.splitext(origins.get(filename, filename))[0])
        display_stem = remove_copy_suffix(os.path.splitext(filename)[0])
        possible_keys = [source_stem, name_to_slug(display_stem)]
        key = next((candidate for candidate in possible_keys if candidate in catalog), None)
        if key is None:
            unmatched.append(filename)
            continue

        targets = catalog[key]
        if len(targets) == 1:
            by_target[targets[0]].append(filename)
            continue

        # A current human-readable filename can disambiguate a future key
        # collision.  A raw hashed filename cannot, so it is safely skipped.
        matching_targets = [
            target for target in targets
            if os.path.splitext(sanitize(target))[0].casefold() == display_stem.casefold()
        ]
        if len(matching_targets) == 1:
            by_target[matching_targets[0]].append(filename)
        else:
            ambiguous.append({"file": filename, "key": key, "targets": targets})

    existing = set(files)
    taken: set[str] = set()
    plan: list[dict] = []
    for photo_name in sorted(by_target):
        target_stem, target_ext = os.path.splitext(sanitize(photo_name))
        group_files = sorted(by_target[photo_name])

        # Keep already valid target names in place. This is the key to an
        # idempotent second run and also preserves old duplicate numbering.
        remaining = []
        for filename in group_files:
            stem, _ = os.path.splitext(filename)
            base = remove_copy_suffix(stem)
            if base.casefold() == target_stem.casefold():
                taken.add(filename)
                copy_match = re.search(r" \((\d+)\)$", stem)
                plan.append({
                    "src": filename,
                    "dst": filename,
                    "csv_name": photo_name,
                    "duplicate_index": int(copy_match.group(1)) - 1 if copy_match else 0,
                })
            else:
                remaining.append(filename)

        for filename in remaining:
            # Расширение берём фактическое: в дампе встречаются .jpg/.png там,
            # где CSV обещает .webp. Совпадает в подавляющем большинстве случаев.
            src_ext = os.path.splitext(filename)[1] or target_ext
            candidate = f"{target_stem}{src_ext}"
            # Один ключ может быть у нескольких загрузок (разный hex) — второй
            # и далее получают детерминированный суффикс, чтобы ничего не потерять.
            bump = 2
            while candidate in taken or (candidate in existing and candidate not in group_files):
                candidate = f"{target_stem} ({bump}){src_ext}"
                bump += 1
            taken.add(candidate)
            plan.append(
                {
                    "src": filename,
                    "dst": candidate,
                    "csv_name": photo_name,
                    "duplicate_index": bump - 2 if bump > 2 else 0,
                }
            )
    return sorted(plan, key=lambda item: item["src"]), unmatched, ambiguous


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


def merge_journal(journal_path: Path, done: list[dict]) -> list[dict]:
    """Preserve raw origins while advancing their current destination names."""
    previous = []
    if journal_path.exists():
        previous = json.loads(journal_path.read_text(encoding="utf-8")).get("renames", [])
    moved = {record["src"]: record["dst"] for record in done}
    merged = []
    known_current = set()
    for record in previous:
        current = record["dst"]
        final = moved.get(current, current)
        merged.append({"src": record["src"], "dst": final})
        known_current.add(current)
    for record in done:
        if record["src"] not in known_current:
            merged.append(record)
    return sorted(merged, key=lambda record: (record["src"], record["dst"]))


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
    parser.add_argument("--check", action="store_true", help="ошибка, если нужны переименования или есть неоднозначность")
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
    plan, unmatched, ambiguous = plan_renames(args.img_dir, catalog, args.journal)
    total_files = len(plan) + len(unmatched) + len(ambiguous)
    changes = [item for item in plan if item["src"] != item["dst"]]

    shown = changes if args.limit == 0 else changes[: args.limit]
    for item in shown:
        mark = " [дубль]" if item["duplicate_index"] else ""
        print(f"{item['src']}  ->  {item['dst']}{mark}")
    if len(shown) < len(changes):
        print(f"... и ещё {len(changes) - len(shown)} (--limit 0, чтобы показать все)")

    print()
    print(f"CSV: уникальных «Название фото» — {sum(len(values) for values in catalog.values())}")
    print(f"Файлов в каталоге: {total_files}")
    print(f"Сопоставлено: {len(plan)}")
    print(f"Уже названы правильно: {len(plan) - len(changes)}")
    print(f"Нужно переименовать: {len(changes)}")
    print(f"Неоднозначных (безопасно пропущены): {len(ambiguous)}")
    print(f"Без совпадения (останутся как есть): {len(unmatched)}")

    if args.check:
        if changes or ambiguous:
            print("\nПроверка не пройдена: состояние не идемпотентно или есть неоднозначные ключи.", file=sys.stderr)
            return 1
        print("\nПроверка пройдена: повторный запуск ничего не переименует.")
        return 0

    if not args.apply:
        print("\nСухой прогон. Запустите с --apply, чтобы переименовать.")
        return 0

    done = apply_renames(args.img_dir, changes)
    journal_records = merge_journal(args.journal, done)
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    args.journal.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "img_dir": str(args.img_dir),
                "rule": "exact Strapi nameToSlug key; ambiguous keys are skipped",
                "renames": journal_records,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"\nПереименовано: {len(done)}")
    print(f"Журнал: {args.journal} (откат: --revert {args.journal})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
