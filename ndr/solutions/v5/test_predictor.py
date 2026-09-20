"""No-network contract tests for the v5 one-shot predictor."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from ndr.solutions.v5 import models
from ndr.solutions.v5 import predictor


class V5PredictorTests(unittest.TestCase):
    def test_output_slug_is_limited_to_request_group(self) -> None:
        output_model = models.nearest_output_model(["first", "second"])
        payload = {
            "checklist": {
                "maker": {"observation": "maker"},
                "profile": {"observation": "profile"},
                "year": {"observation": "year"},
            },
            "slug": "third",
        }
        with self.assertRaises(ValidationError):
            output_model.model_validate(payload)

    def test_output_allows_not_found(self) -> None:
        output_model = models.nearest_output_model(["first", "second"])
        payload = {
            "checklist": {
                "maker": {"observation": "maker is unknown"},
                "profile": {"observation": "all profiles conflict"},
                "year": {"observation": "year is not decisive"},
            },
            "slug": models.NOT_FOUND,
        }
        self.assertEqual(
            output_model.model_validate(payload).slug,
            models.NOT_FOUND,
        )

    def test_system_message_contains_exact_schema_with_not_found(self) -> None:
        output_model = models.nearest_output_model(["first", "second"])
        output_format = models.response_format("ndr_select_nearest", output_model)
        system_prompt = predictor.system_prompt_with_schema("SELECT", output_format)

        self.assertIn('"enum": [', system_prompt)
        self.assertIn('"not_found"', system_prompt)
        self.assertIn('"additionalProperties": false', system_prompt)

    def test_mixed_group_gets_per_element_year_rules(self) -> None:
        prompts = {
            "resolve_multiple_same": {"content": "SELECT"},
            "compare_year_matters": {"content": "EXACT"},
            "compare_year_not_matter": {"content": "IGNORE"},
        }
        candidates = [
            {"slug": "with-year", "vintage": "2019"},
            {"slug": "without-year", "vintage": ""},
        ]
        system_prompt, policies = predictor.selection_system_prompt(
            prompts,
            candidates,
        )
        self.assertIn("EXACT", system_prompt)
        self.assertIn("IGNORE", system_prompt)
        self.assertIn(
            "ELEMENT 1 (with-year): an exact match with 2019 is required",
            system_prompt,
        )
        self.assertIn(
            "ELEMENT 2 (without-year): an exact vintage match is not required",
            system_prompt,
        )
        self.assertEqual(
            [policy["exact_year_required"] for policy in policies],
            [True, False],
        )

    def test_predict_makes_one_selection_call_and_can_return_not_found(self) -> None:
        png_signature = b"\x89PNG\r\n\x1a\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            query = root / "query.png"
            first = root / "first.png"
            second = root / "second.png"
            for path in (query, first, second):
                path.write_bytes(png_signature)

            request = {
                "query": {"image_path": str(query)},
                "group": {
                    "candidates": [
                        {
                            "slug": "first",
                            "vintage": "2019",
                            "reference_image_path": str(first),
                        },
                        {
                            "slug": "second",
                            "vintage": "",
                            "reference_image_path": str(second),
                        },
                    ],
                },
                "prompts": {
                    "resolve_multiple_same": {"content": "SELECT"},
                    "compare_year_matters": {"content": "EXACT"},
                    "compare_year_not_matter": {"content": "IGNORE"},
                },
                "runtime": {
                    "model": "test/model",
                    "api_key_env": "V5_TEST_API_KEY",
                    "image_detail": "original",
                },
            }

            selected = {
                "checklist": {
                    "maker": {"observation": "maker"},
                    "profile": {"observation": "profile"},
                    "year": {"observation": "2019"},
                },
                "slug": models.NOT_FOUND,
            }

            with (
                patch.dict(os.environ, {"V5_TEST_API_KEY": "secret"}),
                patch.object(
                    predictor,
                    "call_model",
                    return_value={
                        "status": "ok",
                        "error": None,
                        "selected": selected,
                        "trace": {},
                    },
                ) as call_model,
            ):
                result = predictor.predict(request)

        self.assertEqual(result["slug"], models.NOT_FOUND)
        self.assertEqual(result["_status"], "ok")
        self.assertEqual(result["_trace"]["pipeline"], "select_nearest_once")
        self.assertNotIn("resolution_call", result["_trace"])
        self.assertEqual(call_model.call_count, 1)
        output_schema = call_model.call_args.kwargs["output_format"]
        slug_schema = output_schema["json_schema"]["schema"]["properties"]["slug"]
        self.assertEqual(
            slug_schema["enum"],
            ["first", "second", models.NOT_FOUND],
        )


if __name__ == "__main__":
    unittest.main()
