"""Check that the included v2 package has its required prompt and knowledge files."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.main import app
from app.solution.v2 import prepare_request, respond


class FakeHTTPResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        first_content = json.dumps({
            "content": "Кокур — белое вино.\nЧем я могу помочь?",
            "suggestions": ["Что во вкусе?", "Как подавать?"],
        }, ensure_ascii=False)
        return json.dumps({"choices": [{"message": {"content": first_content}}]}, ensure_ascii=False).encode()


class SolutionBundleTests(unittest.TestCase):
    def test_secret_file_takes_precedence_and_key_stays_out_of_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            secret = Path(directory) / "openrouter_key"
            secret.write_text("  file-secret\n", encoding="utf-8-sig")
            with (
                patch.dict(os.environ, {
                    "OPENROUTER_API_KEY": "env-secret",
                    "OPENROUTER_API_KEY_FILE": str(secret),
                }),
                patch("app.solution.v2.sommelier.urllib.request.urlopen", return_value=FakeHTTPResponse()) as send,
            ):
                result = respond({"wine": {"Название вина": "Кокур"}, "candidates": [], "history": []})

        self.assertEqual(result["_status"], "ok")
        self.assertEqual(send.call_args.args[0].get_header("Authorization"), "Bearer file-secret")
        self.assertNotIn("file-secret", json.dumps(result["_trace"]))
        self.assertNotIn("env-secret", json.dumps(result["_trace"]))

    def test_openapi_snapshot_is_current(self) -> None:
        snapshot = Path(__file__).resolve().parent.parent / "docs" / "openapi.json"
        self.assertEqual(json.loads(snapshot.read_text(encoding="utf-8")), app.openapi())

    def test_first_and_follow_up_requests_use_bundled_resources(self) -> None:
        wine = {"Название вина": "Кокур", "Категория": "Белое"}
        candidate = {"Название вина": "Другой кокур"}
        first = prepare_request({"wine": wine, "candidates": [candidate], "history": []})
        self.assertTrue(first.first_turn)
        self.assertIn("Кокур", first.payload["messages"][0]["content"])
        self.assertIn("response_format", first.payload)
        follow_up = prepare_request({
            "wine": wine,
            "candidates": [candidate],
            "history": [
                {"role": "assistant", "content": "Первый ответ"},
                {"role": "user", "content": "Какая альтернатива?"},
            ],
        })
        self.assertFalse(follow_up.first_turn)
        self.assertIn("Другой кокур", follow_up.payload["messages"][0]["content"])
        self.assertNotIn("response_format", follow_up.payload)


if __name__ == "__main__":
    unittest.main()
