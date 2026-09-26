#!/usr/bin/env python3
"""Make contact sheets for visual review of a near-duplicate candidate CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "technical" / "strapi" / "img"
TILE_WIDTH = 480
IMAGE_HEIGHT = 510
HEADER_HEIGHT = 92
ROW_HEIGHT = HEADER_HEIGHT + IMAGE_HEIGHT


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def fit_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        source.load()
        rgba = source.convert("RGBA")
        if source.mode == "RGBA":
            bounds = rgba.getchannel("A").getbbox()
            if bounds:
                rgba = rgba.crop(bounds)
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        background.thumbnail((TILE_WIDTH - 20, IMAGE_HEIGHT - 12), Image.Resampling.LANCZOS)
        return background


def draw_page(rows: list[dict[str, str]], page: int, output: Path) -> None:
    canvas = Image.new("RGB", (TILE_WIDTH * 2, ROW_HEIGHT * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, small_font = font(19), font(14)
    for number, row in enumerate(rows):
        top = number * ROW_HEIGHT
        draw.rectangle((0, top, TILE_WIDTH * 2 - 1, top + ROW_HEIGHT - 1), outline="#bbbbbb")
        candidate_id = row.get("review_id") or row.get("candidate_number") or "?"
        score = row.get("edge_similarity", "")
        draw.text((10, top + 5), f"{page}:{number + 1}  ID {candidate_id}  edge {score}", fill="black", font=title_font)
        for side in (1, 2):
            x = (side - 1) * TILE_WIDTH
            slug = row[f"slug_{side}"]
            draw.text((x + 10, top + 34), slug[:60], fill="#333333", font=small_font)
            if len(slug) > 60:
                draw.text((x + 10, top + 53), slug[60:120], fill="#333333", font=small_font)
            filename = row.get(f"image_{side}", "")
            if filename:
                path = IMAGE_DIR / filename
                if path.exists():
                    picture = fit_image(path)
                    canvas.paste(picture, (x + (TILE_WIDTH - picture.width) // 2, top + HEADER_HEIGHT))
                else:
                    draw.text((x + 10, top + HEADER_HEIGHT), "IMAGE MISSING", fill="red", font=title_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pairs-per-sheet", type=int, default=4)
    parser.add_argument("--start", type=int, default=0, help="zero-based row offset")
    parser.add_argument("--limit", type=int, default=0, help="0 means all remaining rows")
    args = parser.parse_args()
    if args.pairs_per_sheet < 1:
        parser.error("--pairs-per-sheet must be positive")
    with args.csv.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    rows = rows[args.start:args.start + args.limit if args.limit else None]
    for page in range(math.ceil(len(rows) / args.pairs_per_sheet)):
        chunk = rows[page * args.pairs_per_sheet:(page + 1) * args.pairs_per_sheet]
        draw_page(chunk, page + 1 + args.start // args.pairs_per_sheet,
                  args.output_dir / f"page_{page + 1 + args.start // args.pairs_per_sheet:04d}.jpg")
    print(f"{len(rows)} pairs, {math.ceil(len(rows) / args.pairs_per_sheet)} sheets -> {args.output_dir}")


if __name__ == "__main__":
    main()
