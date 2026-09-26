#!/usr/bin/env python3
"""Compare every gallery bottle image with its current Catalog reference."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

import near_duplicates as nd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/near_duplicates"
SIZE = (128, 384)
MAX_MEAN_CHANNEL_DIFF = 8.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    catalog = {r["Slug"]: r for r in read_csv(ROOT / "data/technical/strapi/catalog_dataset.csv")}
    registry = read_csv(DATA / "all_candidates.csv")
    folders = {int(p.name.split("_", 1)[0]): p for p in (DATA / "C-visually-close").iterdir() if p.is_dir()}
    bank = nd.ImageBank([])
    current: dict[str, Image.Image] = {}
    mismatches = []
    maximum = (0.0, "")
    checked = 0
    for row in registry:
        number = int(row["candidate_number"])
        folder = folders[number]
        for side in (1, 2):
            slug = row[f"slug_{side}"]
            if slug not in current:
                filename = catalog[slug]["Название фото"]
                current[slug] = bank.normalized(filename, SIZE[0]).resize(SIZE, Image.Resampling.LANCZOS)
            pngs = [p for p in folder.iterdir() if re.match(rf"^{side}_.+\.png$", p.name)]
            if len(pngs) != 1:
                raise ValueError(f"candidate {number}: side {side} has {len(pngs)} bottle PNGs")
            with Image.open(pngs[0]) as source:
                saved = source.convert("RGB").resize(SIZE, Image.Resampling.LANCZOS)
            value = sum(ImageStat.Stat(ImageChops.difference(saved, current[slug])).mean) / 3
            checked += 1
            if value > maximum[0]:
                maximum = (value, f"{number}:{side} {slug}")
            if value > MAX_MEAN_CHANNEL_DIFF:
                mismatches.append((number, side, slug, round(value, 2)))
    print(f"Compared {checked} gallery bottle PNGs with current references; maximum mean-channel difference: {maximum}")
    print(f"Above threshold {MAX_MEAN_CHANNEL_DIFF}: {len(mismatches)}")
    for item in mismatches[:30]:
        print(f"MISMATCH {item}", file=sys.stderr)
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
