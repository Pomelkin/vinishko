"""Make contact sheets for manual review of the saved dense-search top five."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "technical/.new_test_retrieval_cache/review"
FONT_PATH = "C:/Windows/Fonts/arial.ttf"


def thumbnail(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(background, image).convert("RGB")
        image.thumbnail((width, height))
        return image


def main() -> None:
    with (DATA / "technical" / "new_test_top5.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_query: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_query[row["query_file"]].append(row)
    queries = sorted(by_query)
    if any(len(by_query[name]) != 5 for name in queries):
        raise ValueError("Each photo must have exactly five candidates")

    provisional = json.loads((DATA / "technical/.new_test_retrieval_cache/provisional_labels.json").read_text(encoding="utf-8"))
    all_queries = sorted((DATA / "legacy" / "new_test").glob("*.webp"))
    provisional_by_name = {path.name: provisional[str(index)] for index, path in enumerate(all_queries)}
    if set(queries) != set(provisional_by_name):
        raise ValueError("Top-five and provisional query sets differ")

    OUT.mkdir(parents=True, exist_ok=True)
    header_font = ImageFont.truetype(FONT_PATH, 18)
    detail_font = ImageFont.truetype(FONT_PATH, 15)
    cell_w, row_h, image_h = 250, 360, 245
    for page_start in range(0, len(queries), 10):
        page = Image.new("RGB", (cell_w * 6, row_h * 10), "white")
        draw = ImageDraw.Draw(page)
        for local_row, query_name in enumerate(queries[page_start : page_start + 10]):
            items = by_query[query_name]
            for col in range(6):
                x0, y0 = col * cell_w, local_row * row_h
                draw.rectangle((x0, y0, x0 + cell_w - 1, y0 + row_h - 1), outline="#dddddd", width=1)
                if col == 0:
                    image_path = DATA / "legacy" / "new_test" / query_name
                    title = f"{page_start + local_row:03d} {query_name[:18]}"
                    detail_lines = ["query", provisional_by_name[query_name][:31], provisional_by_name[query_name][31:62]]
                else:
                    row = items[col - 1]
                    image_path = DATA / "technical/strapi/img" / row["catalog_image"]
                    title = f"#{col} cos {float(row['cosine_similarity']):.3f}"
                    detail_lines = [row["slug"][:31], row["slug"][31:62], row["wine_name"][:31], f"{row['vintage']} {row['sugar']}".strip()]
                draw.text((x0 + 5, y0 + 3), title, font=header_font, fill="black")
                thumb = thumbnail(image_path, cell_w - 10, image_h)
                page.paste(thumb, (x0 + (cell_w - thumb.width) // 2, y0 + 27 + (image_h - thumb.height) // 2))
                for detail_index, detail in enumerate(detail_lines):
                    draw.text((x0 + 5, y0 + 277 + detail_index * 18), detail, font=detail_font, fill="black")
        page.save(OUT / f"review_{page_start // 10:02d}.jpg", quality=88)
    print(f"Saved {len(queries) // 10} review sheets to {OUT}")


if __name__ == "__main__":
    main()
