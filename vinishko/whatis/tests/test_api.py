"""HTTP contract checks without calling the provider."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from httpx import ASGITransport
from httpx import AsyncClient

from vinishko.whatis.router import MAX_IMAGE_BYTES
from vinishko.whatis.router import router
from vinishko.whatis.service import RecognitionService
from vinishko.whatis.solution.catalog import UNKNOWN


JPEG = b"\xff\xd8\xff\xe0test-image"


def create_app(service: RecognitionService | None = None) -> FastAPI:
    """Only the whatis route, without the recognition pipeline of the main application."""
    app = FastAPI()
    app.include_router(router)
    app.state.whatis = service if service is not None else RecognitionService()
    return app


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
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/whatis", files={"image": ("bottle.jpg", JPEG, "image/jpeg")}
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"category": "Красное", "brand": UNKNOWN})
        self.assertEqual(seen, [JPEG])

    async def test_rejects_large_and_unsupported_images(self) -> None:
        app = create_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            too_large = await client.post(
                "/whatis",
                files={
                    "image": ("large.jpg", JPEG + b"x" * MAX_IMAGE_BYTES, "image/jpeg")
                },
            )
            unsupported = await client.post(
                "/whatis",
                files={"image": ("bad.jpg", b"not an image", "image/jpeg")},
            )

        self.assertEqual(too_large.status_code, 413)
        self.assertEqual(unsupported.status_code, 415)

    async def test_missing_key_and_provider_error_are_not_abstentions(self) -> None:
        app = create_app()
        with patch.dict(
            os.environ, {"OPENROUTER_API_KEY": "", "OPENROUTER_API_KEY_FILE": ""}
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                missing_key = await client.post(
                    "/whatis",
                    files={"image": ("bottle.jpg", JPEG, "image/jpeg")},
                )
        failed_app = create_app(
            RecognitionService(
                lambda _image: {
                    "category": UNKNOWN,
                    "brand": UNKNOWN,
                    "_status": "api_error",
                    "_error": "upstream failed",
                }
            )
        )
        async with AsyncClient(
            transport=ASGITransport(app=failed_app), base_url="http://test"
        ) as client:
            provider_error = await client.post(
                "/whatis",
                files={"image": ("bottle.jpg", JPEG, "image/jpeg")},
            )

        self.assertEqual(missing_key.status_code, 503)
        self.assertEqual(provider_error.status_code, 502)
        self.assertNotIn("upstream failed", provider_error.text)
