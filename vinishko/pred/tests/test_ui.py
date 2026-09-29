"""Проверки серверного адаптера без моделей и внешних API."""

import csv
import unittest
from pathlib import Path

from vinishko.pred.schemas import (
    BottleOut,
    CandidateOut,
    ImageOut,
    MatchOut,
    RecognizeResponse,
)
from vinishko.pred.ui import WineCatalog, number, recognition_for_ui


class UiAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        with (Path(__file__).parents[1] / "catalog.csv").open(
            encoding="utf-8-sig", newline=""
        ) as source:
            self.rows = {row["Slug"]: row for row in csv.DictReader(source)}
        self.slug = next(iter(self.rows))
        self.catalog = WineCatalog(self.rows, {self.slug: "actual-image.webp"})

    def test_real_catalog_preserves_missing_values(self) -> None:
        self.assertEqual(len(self.catalog.search("")), len(self.rows))
        card = self.catalog.details(self.slug)
        self.assertEqual(card["catalog"], self.rows[self.slug])
        self.assertEqual(card["imageUrl"], "/api/catalog/images/actual-image.webp")
        self.assertIsNone(card["ratings"]["community"])
        self.assertIsNone(card["servingTemperature"])
        self.assertTrue(self.catalog.search(card["name"].upper()))
        with self.assertRaises(KeyError):
            self.catalog.details("absent-wine")

    def test_no_invented_abv(self) -> None:
        self.assertIsNone(number({"abv": "135"}, "abv", 100))
        self.assertIsNone(number({"abv": "12–14"}, "abv", 100))
        self.assertEqual(number({"abv": "12,5%"}, "abv", 100), 12.5)

    def test_geometry_candidates_and_metrics(self) -> None:
        other = list(self.rows)[1]
        candidate = CandidateOut(
            slug=other,
            score=0.8,
            group="group",
            image_url="/catalog/images/other.jpg",
            catalog=self.rows[other],
        )
        match = MatchOut(
            slug=self.slug,
            score=0.9,
            group="group",
            image_url="/catalog/images/one.jpg",
            catalog=self.rows[self.slug],
            source="final",
        )
        response = RecognizeResponse(
            image=ImageOut(width=200, height=100),
            bottles=[
                BottleOut(
                    uuid="bottle",
                    index=1,
                    score=0.95,
                    polygons=[
                        [[20, 10], [100, 10], [100, 90], [20, 90]],
                        [[110, 10], [120, 10], [120, 20]],
                    ],
                    label_polygons=None,
                    bbox=[20, 10, 120, 90],
                    status="matched",
                    match=match,
                    candidates=[candidate],
                ),
            ],
            ignored=2,
            timings_s={"normalization": 0.5, "search": 0.1},
        )
        result = recognition_for_ui(response, self.catalog)
        bottle = result["detections"][0]
        self.assertEqual(bottle["polygon"][0], [0.1, 0.1])
        self.assertEqual(len(bottle["polygons"]), 2)
        self.assertEqual(bottle["similar"][0]["slug"], other)
        self.assertIsNone(bottle["match"]["matchConfidence"])
        self.assertFalse(result["metrics"]["isMock"])
        self.assertIsNone(result["metrics"]["f1Top1"])
        self.assertEqual(result["processingTimeMs"], 600)
        response.bottles[0].status = "rejected"
        response.bottles[0].match = None
        rejected = recognition_for_ui(response, self.catalog)
        self.assertIsNone(rejected["bestMatch"])
        self.assertEqual(rejected["detections"][0]["status"], "unmatched")
        response.bottles = []
        self.assertEqual(recognition_for_ui(response, self.catalog)["detections"], [])


if __name__ == "__main__":
    unittest.main()
