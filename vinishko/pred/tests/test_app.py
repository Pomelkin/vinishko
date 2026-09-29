"""HTTP-проверки собранного приложения с отключёнными моделями."""

import unittest
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fastapi.testclient import TestClient
from PIL import Image

from vinishko.app import create_app
from vinishko.pred.pipeline.pipeline import Pipeline, PipelineResult
from vinishko.pred.pipeline.structs import (
    BottleCrop,
    BottleCandidates,
    Candidate,
    MatchedBottle,
)
from vinishko.sommelier.service import SommelierService
from vinishko.sommelier.storage import JsonSessionStore
from vinishko.sommelier.tests.test_api import FakeEngine, SESSION_ID
from vinishko.whatis.service import RecognitionService


class ApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Настоящий оркестратор и каталог, без загрузки модели и внешних сервисов.
        cls.pipeline = Pipeline(
            normalizer=lambda image: [], search=False, resolve=False
        )

    def test_catalog_health_and_recognition_routes(self) -> None:
        with TestClient(create_app(self.pipeline)) as client:
            self.assertEqual(client.get("/api/health").status_code, 200)
            cards = client.get("/api/wines").json()
            self.assertEqual(len(cards), len(self.pipeline.catalog))
            slug = "perovskih_aligote"
            details = client.get(f"/api/wines/{slug}")
            self.assertEqual(details.status_code, 200)
            self.assertEqual(details.json()["catalog"]["Slug"], slug)
            self.assertTrue(
                client.get("/api/wines", params={"q": details.json()["name"]}).json()
            )
            self.assertEqual(client.get("/api/wines/no-such-wine").status_code, 404)
            picture = BytesIO()
            Image.new("RGB", (32, 48), "white").save(picture, "JPEG")
            for route in ["/api/recognize", "/recognize"]:
                response = client.post(
                    route,
                    files={"image": ("photo.jpg", picture.getvalue(), "image/jpeg")},
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["image"]["width"], 32)
                self.assertEqual(
                    response.json()[
                        "detections" if route.startswith("/api/") else "bottles"
                    ],
                    [],
                )
                self.assertEqual(
                    client.post(
                        route, files={"image": ("bad.jpg", b"invalid", "image/jpeg")}
                    ).status_code,
                    400,
                )
            self.assertEqual(
                client.get("/api/catalog/images/none.jpg").status_code, 404
            )
            self.assertEqual(client.get("/api/nonexistent").status_code, 404)

    def test_matched_result_keeps_search_candidates_for_ui(self) -> None:
        pixels = np.zeros((48, 32, 3), dtype=np.uint8)
        crop = BottleCrop(
            1,
            0.95,
            [[[1, 1], [30, 1], [30, 46], [1, 46]]],
            [],
            0,
            pixels,
            {},
            pixels,
            {},
        )
        slugs = list(self.pipeline.catalog.rows)[:2]
        candidates = [
            Candidate(slug, 0.9, pixels, crop, slug, True, {"image": f"{slug}.jpg"}, {})
            for slug in slugs
        ]
        result = PipelineResult(
            [crop],
            [BottleCandidates(crop, candidates)],
            [MatchedBottle(crop, candidates[0], "final", {})],
        )
        picture = BytesIO()
        Image.fromarray(pixels).save(picture, "JPEG")
        with (
            patch.object(Pipeline, "__call__", return_value=result),
            TestClient(create_app(self.pipeline)) as client,
        ):
            upload = {"image": ("bottle.jpg", picture.getvalue(), "image/jpeg")}
            response = client.post("/api/recognize", files=upload)
            self.assertEqual(response.status_code, 200, response.text)
            bottle = response.json()["detections"][0]
            self.assertEqual(bottle["match"]["slug"], slugs[0])
            self.assertEqual([c["slug"] for c in bottle["similar"]], [slugs[1]])
            self.assertEqual(bottle["polygon"][0], [1 / 32, 1 / 48])
            raw = client.post("/recognize", files=upload).json()["bottles"][0]
            self.assertEqual(raw["candidates"], [])

    def test_chat_and_whatis_aliases(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            TestClient(create_app(self.pipeline)) as client,
        ):
            client.app.state.sommelier = SommelierService(
                JsonSessionStore(Path(directory)), FakeEngine()
            )
            client.app.state.whatis = RecognitionService(
                lambda data: {
                    "_status": "ok",
                    "category": "Красное",
                    "brand": "Фанагория",
                }
            )
            wine = client.get("/api/wines/perovskih_aligote").json()
            route = f"/api/sommelier/sessions/{SESSION_ID}"
            opening = client.post(
                route,
                json={"wine": wine["catalog"], "candidates": wine["candidateCatalogs"]},
            )
            self.assertEqual(opening.status_code, 201, opening.text)
            self.assertEqual(
                client.post(
                    route + "/messages", json={"content": "Как подавать?"}
                ).status_code,
                201,
            )
            self.assertEqual(len(client.get(route).json()["messages"]), 3)
            # Исходный маршрут видит ту же сохранённую сессию.
            self.assertEqual(client.get(route.removeprefix("/api")).status_code, 200)
            self.assertEqual(client.delete(route).status_code, 204)
            self.assertEqual(client.get(route).status_code, 404)
            photo = BytesIO()
            Image.new("RGB", (16, 16)).save(photo, "JPEG")
            for path in ["/api/whatis", "/whatis"]:
                response = client.post(
                    path,
                    files={"image": ("bottle.jpg", photo.getvalue(), "image/jpeg")},
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["brand"], "Фанагория")

    def test_spa(self) -> None:
        if not Path("frontend/dist/index.html").is_file():
            self.skipTest("Production frontend не собран")
        with TestClient(create_app(self.pipeline)) as client:
            for path in ["/", "/scan/sample-id", "/wine/perovskih_aligote"]:
                response = client.get(path)
                self.assertEqual(response.status_code, 200, path)
                self.assertIn("text/html", response.headers["content-type"])


if __name__ == "__main__":
    unittest.main()
