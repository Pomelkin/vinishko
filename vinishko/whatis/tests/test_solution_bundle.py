"""The copied prompt, settings, and catalog remain wired into the model request."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vinishko.whatis.solution.catalog import UNKNOWN
from vinishko.whatis.solution.catalog import catalog_values
from vinishko.whatis.solution.predictor import predict


JPEG = b"\xff\xd8\xff\xe0test-image"


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.body = json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self.body


class SolutionTests(unittest.TestCase):
    def test_predict_uses_bundled_lists_and_original_model_settings(self) -> None:
        categories, brands = catalog_values()
        self.assertEqual(len(categories), 4)
        self.assertEqual(len(brands), 135)
        provider = FakeResponse(
            json.dumps(
                {"category": categories[0], "brand": brands[0]}, ensure_ascii=False
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            key_file = Path(directory) / "key.txt"
            key_file.write_text("file-secret\n", encoding="utf-8")
            with (
                patch.dict(
                    os.environ,
                    {
                        "OPENROUTER_API_KEY": "env-secret",
                        "OPENROUTER_API_KEY_FILE": str(key_file),
                    },
                ),
                patch(
                    "vinishko.whatis.solution.predictor.urllib.request.OpenerDirector.open",
                    return_value=provider,
                ) as send,
            ):
                result = predict(JPEG)

        self.assertEqual(result["_status"], "ok")
        request = send.call_args.args[0]
        payload = json.loads(request.data)
        schema = payload["response_format"]["json_schema"]["schema"]
        self.assertEqual(request.get_header("Authorization"), "Bearer file-secret")
        self.assertEqual(
            set(schema["properties"]["category"]["enum"]), {*categories, UNKNOWN}
        )
        self.assertEqual(set(schema["properties"]["brand"]["enum"]), {*brands, UNKNOWN})
        self.assertEqual(payload["model"], "deepseek/deepseek-v4.1-flash")
        self.assertEqual(payload["reasoning"]["effort"], "none")
        self.assertEqual(payload["provider"]["only"], ["together"])
        self.assertTrue(
            payload["messages"][1]["content"][1]["image_url"]["url"].startswith(
                "data:image/jpeg;base64,"
            )
        )
        self.assertNotIn("file-secret", json.dumps(result["_trace"]))

    def test_invalid_catalog_value_is_an_error(self) -> None:
        response = FakeResponse('{"category":"Красное","brand":"Unknown Winery"}')
        with (
            patch.dict(
                os.environ,
                {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_API_KEY_FILE": ""},
            ),
            patch(
                "vinishko.whatis.solution.predictor.urllib.request.OpenerDirector.open",
                return_value=response,
            ),
        ):
            result = predict(JPEG)
        self.assertEqual(result["_status"], "invalid_response")
        self.assertEqual(result["brand"], UNKNOWN)

    def test_openrouter_proxy_is_local_to_request(self) -> None:
        categories, brands = catalog_values()
        for proxy in (" http://proxy.example:8080 ", ""):
            with (
                self.subTest(proxy=proxy),
                patch.dict(
                    os.environ,
                    {
                        "OPENROUTER_API_KEY": "test-key",
                        "openrouter_http_proxy": proxy,
                        "HTTPS_PROXY": "http://unrelated.example:8080",
                    },
                    clear=True,
                ),
                patch(
                    "vinishko.whatis.solution.predictor.urllib.request.build_opener"
                ) as build,
            ):
                build.return_value.open.return_value = FakeResponse(
                    json.dumps({"category": categories[0], "brand": brands[0]})
                )
                result = predict(JPEG)
                self.assertEqual(result["_status"], "ok")
                self.assertEqual(
                    build.call_args.args[0].proxies,
                    {"http": proxy.strip(), "https": proxy.strip()} if proxy else {},
                )
                self.assertEqual(
                    os.environ["HTTPS_PROXY"], "http://unrelated.example:8080"
                )
