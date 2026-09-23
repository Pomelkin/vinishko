"""Contract checks for the image extractor, product pool, and offline eval."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from candidates.catalog import ROOT, UNKNOWN, catalog_values
from candidates.eval import evaluate, load_dataset
from candidates.match import match_products
from candidates.solution.v1.predictor import predict


class FakeResponse:
    def __init__(self, content: str):
        self.body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


class CandidatesTests(unittest.TestCase):
    def test_predict_sends_only_image_and_closed_lists(self):
        gold = load_dataset()[0]
        requests = []

        def open_request(request, timeout):
            requests.append((request, timeout))
            return FakeResponse(json.dumps({
                "category": gold["category"], "brand": gold["brand"]
            }, ensure_ascii=False))

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), \
             patch("urllib.request.urlopen", side_effect=open_request):
            result = predict(ROOT / gold["image_path"])
        self.assertEqual(result["_status"], "ok")
        self.assertEqual(result["category"], gold["category"])
        request, timeout = requests[0]
        payload = json.loads(request.data)
        categories, brands = catalog_values()
        schema = payload["response_format"]["json_schema"]["schema"]
        self.assertEqual(set(schema["properties"]["category"]["enum"]), {*categories, UNKNOWN})
        self.assertEqual(set(schema["properties"]["brand"]["enum"]), {*brands, UNKNOWN})
        self.assertEqual(payload["reasoning"]["effort"], "none")
        self.assertEqual(payload["temperature"], 0)
        self.assertTrue(payload["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertNotIn(gold["image_path"], request.data.decode())
        self.assertNotIn(gold["source_slug"], request.data.decode())
        self.assertGreater(timeout, 0)

    def test_invalid_brand_is_not_accepted_as_abstention(self):
        gold = load_dataset()[0]
        response = FakeResponse('{"category":"Розовое","brand":"Unknown Winery"}')
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}), \
             patch("urllib.request.urlopen", return_value=response):
            result = predict(ROOT / gold["image_path"])
        self.assertEqual(result["_status"], "invalid_response")
        self.assertEqual(result["brand"], UNKNOWN)

    def test_matching_and_partial_eval(self):
        gold = load_dataset()[0]
        exact = match_products({"category": gold["category"], "brand": gold["brand"]})
        self.assertIn(gold["source_slug"], {row["Slug"] for row in exact})
        self.assertTrue(all(row["Категория"] == gold["category"] and row["Винодельня"] == gold["brand"] for row in exact))
        self.assertEqual(match_products({"category": UNKNOWN, "brand": UNKNOWN}), [])
        report = evaluate([gold, load_dataset()[1]], [{
            "id": gold["id"], "prediction": {
                "category": gold["category"], "brand": gold["brand"], "_status": "ok"
            }, "latency_ms": 100,
        }])
        self.assertEqual(report["totals"]["dataset_size"], 2)
        self.assertEqual(report["totals"]["predicted"], 1)
        self.assertEqual(report["totals"]["source_slug_in_candidates"], 1)
        self.assertEqual(report["rates"]["both_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
