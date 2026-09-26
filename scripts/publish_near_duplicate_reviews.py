#!/usr/bin/env python3
"""Publish reviewed near-duplicate pairs and their current-reference galleries.

Run only after validate_near_duplicate_review_decisions.py succeeds. The build is
prepared in a sibling directory and checked before replacing the live registry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

import validate_near_duplicate_review_decisions as review_check
import near_duplicates as nd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "near_duplicates"
IMG = ROOT / "data" / "technical" / "strapi" / "img"
REGISTRY = DATA / "all_candidates.csv"
GALLERY = DATA / "C-visually-close"
DECISIONS = DATA / "reviewed_candidates.csv"
CSV_FIELDS = [
    "candidate_number", "slug_1", "slug_2", "winery_1", "winery_2",
    "review_verdict", "is_confirmed_near_duplicate", "visual_relation",
    "diff_percent", "review_source", "review_id", "review_evidence",
    "review_confidence", "reference_sha256_1", "reference_sha256_2",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pair_key(row: dict[str, str]) -> frozenset[str]:
    return frozenset((row["slug_1"], row["slug_2"]))


def load_catalog() -> dict[str, dict[str, str]]:
    rows = read_csv(ROOT / "data" / "technical" / "strapi" / "catalog_dataset.csv")
    return {row["Slug"]: row for row in rows}


def safe_reference(slug: str, catalog: dict[str, dict[str, str]]) -> str:
    row = catalog[slug]
    if not row["Статус изображения"].startswith("ok_"):
        raise ValueError(f"reference is excluded from CV index: {slug}")
    name = row["Название фото"]
    if not name or not (IMG / name).is_file():
        raise ValueError(f"accepted reference missing: {slug}: {name}")
    return name


def load_reviews() -> tuple[dict[tuple[str, str], dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    expected = review_check.inputs()
    actual: dict[tuple[str, str], dict[str, str]] = {}
    for path in sorted(review_check.BATCHES.glob("*_decisions.csv")):
        for row in read_csv(path):
            key = (row["source"], row["review_id"])
            if key in actual:
                raise ValueError(f"duplicate decision: {key}")
            actual[key] = row
    if actual.keys() != expected.keys():
        raise ValueError(f"review decisions incomplete: {len(expected.keys() - actual.keys())} missing")
    for key, row in actual.items():
        if row["visual_checked"].lower() != "true" or not row["evidence"].strip():
            raise ValueError(f"candidate was not visually reviewed: {key}")
        if (row["slug_1"], row["slug_2"]) != (expected[key]["slug_1"], expected[key]["slug_2"]):
            raise ValueError(f"candidate identity changed: {key}")
    return expected, actual


def esc(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def gallery_name(number: int, row: dict[str, str]) -> str:
    # Keep the original IDs; four digits are needed once the registry passes 999.
    return f"{number:03d}_{row['slug_1'][:26]}-vs-{row['slug_2'][:26]}"


def make_zoom(a: Image.Image, b: Image.Image) -> Image.Image:
    # Rank horizontal strips at a common size, then show the strongest strip
    # with enough context to read the labels. Full bottle views remain alongside.
    height = 700
    small_a = a.resize((180, height))
    small_b = b.resize((180, height))
    diff = ImageChops.difference(small_a, small_b).convert("L")
    scores = []
    for y in range(0, height - 35, 10):
        scores.append((ImageStat.Stat(diff.crop((0, y, 180, y + 35))).mean[0], y))
    peak = max(scores)[1]
    top = max(0, min(height - 245, peak - 105))
    panels = []
    for image in (a, b):
        y0, y1 = round(top / height * image.height), round((top + 245) / height * image.height)
        panel = image.crop((0, y0, image.width, y1))
        panel.thumbnail((640, 900), Image.Resampling.LANCZOS)
        panels.append(panel)
    canvas = Image.new("RGB", (sum(p.width for p in panels) + 16, max(p.height for p in panels)), "white")
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (panels[0].width + 16, 0))
    return canvas


def render(folder: Path, row: dict[str, str], catalog: dict[str, dict[str, str]], bank: nd.ImageBank) -> None:
    folder.mkdir()
    names = [safe_reference(row[f"slug_{n}"], catalog) for n in (1, 2)]
    images = [bank.normalized(name, 640) for name in names]
    for n, image in enumerate(images, 1):
        image.save(folder / f"{n}_{row[f'slug_{n}'][:75]}.png")
    h = min(1100, max(image.height for image in images))
    panels = []
    for image in images:
        panel = image.copy()
        panel.thumbnail((550, h), Image.Resampling.LANCZOS)
        panels.append(panel)
    compare = Image.new("RGB", (sum(p.width for p in panels) + 16, max(p.height for p in panels)), "white")
    compare.paste(panels[0], (0, 0))
    compare.paste(panels[1], (panels[0].width + 16, 0))
    compare.save(folder / "_compare.png")
    make_zoom(*images).save(folder / "_zoom.png")

    lines = [f"# {row['slug_1']}\\|{row['slug_2']}", "", "**Вердикт:** подтверждённая пара разных вин с визуально близкой упаковкой", "", f"**Источник проверки:** {row['review_source']} / {row['review_id']}", "", f"**Наблюдение:** {row['review_evidence']}", "", "## Позиции Каталога", ""]
    for n in (1, 2):
        slug = row[f"slug_{n}"]
        item = catalog[slug]
        lines.extend([f"### {n}. `{slug}`", "", "| Поле | Значение |", "|---|---|"])
        for field in ("Винодельня", "Название вина", "Категория", "Сорт винограда", "Винтаж", "Сахар", "Название фото", "SHA256 изображения", "Статус изображения"):
            lines.append(f"| {field} | {esc(item[field])} |")
        lines.append("")
    (folder / "info.md").write_text("\n".join(lines), encoding="utf-8")


def link_folder(source: Path, target: Path) -> None:
    target.mkdir()
    for path in source.iterdir():
        if not path.is_file():
            raise ValueError(f"unexpected gallery entry: {path}")
        dest = target / path.name
        try:
            os.link(path, dest)
        except OSError:
            shutil.copy2(path, dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="replace live CSV and gallery after staging")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted stage with the same review plan")
    parser.add_argument("--workers", type=int, default=4, help="independent image-rendering workers")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    expected, decisions = load_reviews()
    catalog = load_catalog()
    old = read_csv(REGISTRY)
    old_by_number = {int(row["candidate_number"]): row for row in old}
    bad_legacy = {
        int(key[1]) for key, row in decisions.items()
        if key[0] == "legacy_recheck" and row["verdict"] != "confirmed_near_duplicate"
    }
    retained = [row for number, row in old_by_number.items() if number not in bad_legacy]
    keys = {pair_key(row) for row in retained}
    if len(keys) != len(retained):
        raise ValueError("duplicate pair in old registry")
    new = []
    # 387 was a reviewed exclusion and must remain unused.
    next_number = max(max(old_by_number), 387) + 1
    for key, decision in sorted(decisions.items()):
        if decision["verdict"] != "confirmed_near_duplicate" or key[0] == "legacy_recheck":
            continue
        if pair_key(decision) in keys:
            raise ValueError(f"confirmed pair already in registry: {key}")
        keys.add(pair_key(decision))
        for slug in (decision["slug_1"], decision["slug_2"]):
            safe_reference(slug, catalog)
        row = {
            "candidate_number": str(next_number),
            "slug_1": decision["slug_1"], "slug_2": decision["slug_2"],
            "winery_1": catalog[decision["slug_1"]]["Винодельня"],
            "winery_2": catalog[decision["slug_2"]]["Винодельня"],
            "review_verdict": "confirmed_near_duplicate", "is_confirmed_near_duplicate": "true",
            "visual_relation": "visually_close", "diff_percent": "",
            "review_source": key[0], "review_id": key[1],
            "review_evidence": decision["evidence"],
            "review_confidence": decision.get("confidence", ""),
            "reference_sha256_1": catalog[decision["slug_1"]]["SHA256 изображения"],
            "reference_sha256_2": catalog[decision["slug_2"]]["SHA256 изображения"],
        }
        new.append(row)
        next_number += 1
    old_rechecks = {int(k[1]): d for k, d in decisions.items() if k[0] == "legacy_recheck"}
    for row in retained:
        number = int(row["candidate_number"])
        review = old_rechecks.get(number)
        row.setdefault("review_source", "legacy")
        row.setdefault("review_id", str(number))
        row.setdefault("review_evidence", "")
        row.setdefault("review_confidence", "")
        for side in (1, 2):
            row[f"reference_sha256_{side}"] = catalog[row[f"slug_{side}"]]["SHA256 изображения"]
        if review:
            row.update(review_source="legacy_recheck", review_id=str(number),
                       review_evidence=review["evidence"], review_confidence=review.get("confidence", ""))
    combined = sorted(retained + new, key=lambda row: int(row["candidate_number"]))
    verdicts = Counter(row["verdict"] for row in decisions.values())
    print(f"Reviewed candidates: {len(decisions)}; verdicts: {dict(verdicts)}")
    print(f"Registry: {len(old)} old - {len(bad_legacy)} excluded + {len(new)} new = {len(combined)} confirmed")
    if not args.apply:
        return 0

    stage = DATA / ".gallery_review_stage"
    stage_csv = DATA / ".registry_review_stage.csv"
    stage_decisions = DATA / ".reviewed_candidates_stage.csv"
    backup = DATA / ".gallery_before_reviews"
    backup_csv = DATA / ".registry_before_reviews.csv"
    if GALLERY.resolve().parent != DATA.resolve() or stage.resolve().parent != DATA.resolve() or backup.resolve().parent != DATA.resolve():
        raise ValueError("gallery stage/backup path escapes the near-duplicate data directory")
    plan_digest = hashlib.sha256(json.dumps(combined, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    for path in (backup, backup_csv):
        if path.exists():
            raise FileExistsError(f"backup already exists: {path}")
    if args.resume:
        if not stage.is_dir() or (stage / ".plan_sha256").read_text(encoding="ascii") != plan_digest:
            raise ValueError("staging plan differs from current review decisions")
    else:
        for path in (stage, stage_csv, stage_decisions):
            if path.exists():
                raise FileExistsError(f"staging artifact already exists: {path}")
        stage.mkdir()
        (stage / ".plan_sha256").write_text(plan_digest, encoding="ascii")
    existing_folders = {int(p.name.split("_", 1)[0]): p for p in GALLERY.iterdir() if p.is_dir()}
    redraw = {n for n, row in old_by_number.items() if n in old_rechecks and n not in bad_legacy}
    bank = nd.ImageBank([])
    to_render = []
    for row in combined:
        number = int(row["candidate_number"])
        folder = stage / (existing_folders[number].name if number in old_by_number else gallery_name(number, row))
        if folder.exists():
            complete = (folder / "info.md").is_file() and (folder / "_compare.png").is_file() and (folder / "_zoom.png").is_file() and len(list(folder.glob("[12]_*.png"))) == 2
            if complete:
                continue
            if not args.resume or folder.resolve().parent != stage.resolve():
                raise ValueError(f"unexpected incomplete staging folder: {folder}")
            shutil.rmtree(folder)
        if number in old_by_number and number not in redraw:
            link_folder(existing_folders[number], folder)
        else:
            to_render.append((folder, row))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(render, folder, row, catalog, bank) for folder, row in to_render]
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            if done % 100 == 0 or done == len(futures):
                print(f"Rendered {done}/{len(futures)} reviewed galleries", flush=True)
    write_csv(stage_csv, combined, CSV_FIELDS)
    review_rows = []
    for key, row in sorted(decisions.items()):
        review_rows.append({**expected[key], **row})
    fields = list(dict.fromkeys(["source", "review_id", "slug_1", "slug_2", "verdict", "evidence", "visual_checked", "confidence"] + [k for row in review_rows for k in row]))
    write_csv(stage_decisions, review_rows, fields)
    if len([p for p in stage.iterdir() if p.is_dir()]) != len(combined):
        raise ValueError("staged gallery count differs from registry")
    os.replace(GALLERY, backup)
    os.replace(REGISTRY, backup_csv)
    try:
        os.replace(stage, GALLERY)
        os.replace(stage_csv, REGISTRY)
        os.replace(stage_decisions, DECISIONS)
    except Exception:
        if GALLERY.exists():
            os.replace(GALLERY, stage)
        if REGISTRY.exists():
            os.replace(REGISTRY, stage_csv)
        os.replace(backup, GALLERY)
        os.replace(backup_csv, REGISTRY)
        raise
    (GALLERY / ".plan_sha256").unlink()
    print(f"Published {len(combined)} confirmed pairs. Backup retained at {backup} and {backup_csv}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
