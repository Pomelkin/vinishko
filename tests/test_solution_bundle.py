"""Check that the included v2 package has its required prompt and knowledge files."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from ai_somelier.solution.v2 import prepare_request
from sommelier_service.app import app


class SolutionBundleTests(unittest.TestCase):
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
