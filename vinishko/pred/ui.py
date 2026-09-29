"""Адаптер каталога и результатов пайплайна для React-интерфейса."""

import math
import random
import re
from urllib.parse import quote
from uuid import uuid4

from vinishko.pred.schemas import RecognizeResponse
from vinishko.whatis.solution.catalog import UNKNOWN


SUGGESTIONS = 5
"""Сколько вин каталога предлагать бутылке, которой нет в каталоге."""
COLORS_PLURAL = {
    "Белое": "Белые",
    "Красное": "Красные",
    "Розовое": "Розовые",
    "Оранжевое": "Оранжевые",
}


def text_value(row: dict[str, str], key: str) -> str | None:
    value = row.get(key, "").strip()
    return (
        value
        if value and value.casefold() not in {"nan", "не определено", "null"}
        else None
    )


def number(row: dict[str, str], key: str, maximum: float) -> float | None:
    value = text_value(row, key)
    if value is None or not re.fullmatch(r"\d+(?:[.,]\d+)?%?", value):
        return None
    result = float(value.rstrip("%").replace(",", "."))
    return result if 0 <= result <= maximum else None


class WineCatalog:
    """Карточки CSV с именами изображений из payload коллекции."""

    def __init__(self, rows: dict[str, dict[str, str]], images: dict[str, str]) -> None:
        self.rows = rows
        self.summaries = {
            slug: self.summary(slug, row, images.get(slug))
            for slug, row in rows.items()
        }

    @staticmethod
    def summary(slug: str, row: dict[str, str], image: str | None) -> dict:
        vintage = number(row, "Винтаж", 2100)
        return {
            "slug": slug,
            "name": text_value(row, "Название вина") or slug,
            "producer": text_value(row, "Винодельня") or "",
            "imageUrl": f"/api/catalog/images/{quote(image, safe='')}"
            if image
            else None,
            "region": text_value(row, "Регион"),
            "grapeVarieties": [
                v.strip()
                for v in re.split(r"[,;]", row.get("Сорт винограда", ""))
                if v.strip()
            ],
            # «Цвет» в CSV — оттенок; фильтр интерфейса использует категорию цвета.
            "color": text_value(row, "Категория"),
            "category": text_value(row, "Сахар"),
            "vintage": int(vintage)
            if vintage and vintage >= 1900 and vintage.is_integer()
            else None,
            "shortDescription": text_value(row, "Описание"),
            "ratings": {"community": None, "roskachestvo": None},
        }

    def search(self, query: str) -> list[dict]:
        words = query.casefold().replace("ё", "е").split()
        return [
            card
            for card in self.summaries.values()
            if all(
                word
                in " ".join(
                    [
                        card["name"],
                        card["producer"],
                        card["region"] or "",
                        *card["grapeVarieties"],
                    ]
                )
                .casefold()
                .replace("ё", "е")
                for word in words
            )
        ]

    def suggestions(
        self, brand: str | None, category: str | None, seed: str
    ) -> dict | None:
        """Похожие для бутылки, которую отверг поиск, по признакам whatis: вина каталога той же категории, вина той же винодельни — первыми.

        Категорию whatis не узнал — вина его винодельни; не узнал ничего — None. Внутри «своей» винодельни и остальных порядок
        случайный, но для одного seed (id бутылки) один и тот же, позиции с фото — раньше позиций без фото."""
        brand = brand if brand and brand != UNKNOWN else None
        category = category if category and category != UNKNOWN else None
        if category is None and brand is None:
            return None
        rng = random.Random(f"{seed}|{category}|{brand}")

        def shuffled(slugs: list[str]) -> list[str]:
            mixed = rng.sample(slugs, len(slugs))
            return sorted(
                mixed, key=lambda slug: self.summaries[slug]["imageUrl"] is None
            )

        pool = sorted(
            slug
            for slug, row in self.rows.items()
            if (row.get("Категория") or "").strip() == category
            or (category is None and (row.get("Винодельня") or "").strip() == brand)
        )
        own = [
            s for s in pool if (self.rows[s].get("Винодельня") or "").strip() == brand
        ]
        others = [s for s in pool if s not in set(own)]
        picked = (shuffled(own) + shuffled(others))[:SUGGESTIONS]
        if not picked:
            return None
        if category is None:
            title = f"{brand} в каталоге"
        elif own:
            title = f"{category} {brand} в каталоге"
        else:
            title = f"{COLORS_PLURAL.get(category, category)} вина в каталоге"
        return {"title": title, "wines": [self.summaries[slug] for slug in picked]}

    def details(self, slug: str) -> dict:
        row = self.rows[slug]
        group = text_value(row, "near_duplicate_group_slug")
        similar = [
            self.summaries[key]
            for key, other in self.rows.items()
            if key != slug
            and group
            and text_value(other, "near_duplicate_group_slug") == group
        ][:5]
        return {
            **self.summaries[slug],
            "description": text_value(row, "Описание"),
            "alcoholPercent": number(row, "Крепость, %", 100),
            "servingTemperature": None,
            "foodPairings": [],
            "similarWines": similar,
            "catalog": row,
            "candidateCatalogs": [self.rows[card["slug"]] for card in similar],
        }


def area(points: list[list[float]]) -> float:
    return (
        abs(
            sum(
                p[0] * points[(i + 1) % len(points)][1]
                - points[(i + 1) % len(points)][0] * p[1]
                for i, p in enumerate(points)
            )
        )
        / 2
    )


def recognition_for_ui(response: RecognizeResponse, catalog: WineCatalog) -> dict:
    width, height = response.image.width, response.image.height
    detections = []
    for bottle in response.bottles:
        contours = []
        for polygon in bottle.polygons:
            points = [
                [min(1, max(0, x / width)), min(1, max(0, y / height))]
                for x, y in polygon
            ]
            if len(points) >= 3 and area(points) > 1e-8:
                contours.append(points)
        if not contours:
            raise ValueError(f"Пустая маска бутылки {bottle.uuid}")
        matched = bottle.match
        candidates = [
            c for c in bottle.candidates if not matched or c.slug != matched.slug
        ][:5]
        detections.append(
            {
                "id": bottle.uuid,
                "polygon": max(contours, key=area),
                "polygons": contours,
                "detectionConfidence": min(1, max(0, bottle.score))
                if math.isfinite(bottle.score)
                else None,
                "status": "matched" if matched else "unmatched",
                # Косинус поиска не является калиброванной вероятностью совпадения.
                "match": {
                    "slug": matched.slug,
                    "wine": catalog.summaries[matched.slug],
                    "matchConfidence": None,
                }
                if matched
                else None,
                "similar": [
                    {
                        "slug": c.slug,
                        "rank": i + 1,
                        "similarityScore": min(1, max(0, c.score)),
                        "wine": catalog.summaries[c.slug],
                    }
                    for i, c in enumerate(candidates)
                ],
                "rejection": bottle.rejection.model_dump()
                if bottle.rejection
                else None,
                "unknownWine": bottle.unknown_wine.model_dump()
                if bottle.unknown_wine
                else None,
                "unknownWineError": bottle.unknown_wine_error.detail
                if bottle.unknown_wine_error
                else None,
            }
        )
    best = next((d["match"] for d in detections if d["match"]), None)
    return {
        "schemaVersion": "1.0",
        "requestId": str(uuid4()),
        "image": {
            "width": width,
            "height": height,
            "coordinateSpace": "normalized",
            "orientationApplied": True,
        },
        "bestMatch": {"slug": best["slug"]} if best else None,
        "metrics": {
            "f1Top1": None,
            "f1Top5": None,
            "scope": "evaluation-dataset",
            "datasetId": None,
            "isMock": False,
        },
        "detections": detections,
        "ignored": response.ignored,
        "processingTimeMs": round(sum(response.timings_s.values()) * 1000),
    }
