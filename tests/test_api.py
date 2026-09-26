"""HTTP contract checks without calling the provider."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient

from app.main import MAX_IMAGE_BYTES, create_app
from app.service import RecognitionService
from app.solution.catalog import UNKNOWN


JPEG = b"\xff\xd8\xff\xe0test-image"


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_predict_returns_only_public_fields(self) -> None:
        seen: list[bytes] = []

        def engine(image: bytes) -> dict:
            seen.append(image)
            return {
                "category": "Красное",
                "brand": UNKNOWN,
                "_status": "ok",
                "_error": None,
                "_trace": {"raw_response": "internal-only"},
            }

        app = create_app(RecognitionService(engine))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/predict", files={"image": ("bottle.jpg", JPEG, "image/jpeg")})
            health = await client.get("/health")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"category": "Красное", "brand": UNKNOWN})
        self.assertEqual(seen, [JPEG])
        self.assertEqual(health.json(), {"status": "ok"})

    async def test_rejects_large_and_unsupported_images(self) -> None:
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            too_large = await client.post(
                "/v1/predict",
                files={"image": ("large.jpg", JPEG + b"x" * MAX_IMAGE_BYTES, "image/jpeg")},
            )
            unsupported = await client.post(
                "/v1/predict", files={"image": ("bad.jpg", b"not an image", "image/jpeg")},
            )

        self.assertEqual(too_large.status_code, 413)
        self.assertEqual(unsupported.status_code, 415)

    async def test_missing_key_and_provider_error_are_not_abstentions(self) -> None:
        app = create_app()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "OPENROUTER_API_KEY_FILE": ""}):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                missing_key = await client.post(
                    "/v1/predict", files={"image": ("bottle.jpg", JPEG, "image/jpeg")},
                )
        failed_app = create_app(RecognitionService(lambda _image: {
            "category": UNKNOWN, "brand": UNKNOWN, "_status": "api_error", "_error": "upstream failed",
        }))
        async with AsyncClient(transport=ASGITransport(app=failed_app), base_url="http://test") as client:
            provider_error = await client.post(
                "/v1/predict", files={"image": ("bottle.jpg", JPEG, "image/jpeg")},
            )

        self.assertEqual(missing_key.status_code, 503)
        self.assertEqual(provider_error.status_code, 502)
        self.assertNotIn("upstream failed", provider_error.text)


class OpenApiTests(unittest.TestCase):
    def test_snapshot_is_current(self) -> None:
        path = Path(__file__).resolve().parent.parent / "docs" / "openapi.json"
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), create_app().openapi())
