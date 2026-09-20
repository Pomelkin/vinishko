"""No-network contract tests for the v7 image-only predictor."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from ndr.solutions.v7 import models
from ndr.solutions.v7 import predictor


def selected_payload(selection: str) -> dict[str, object]:
    """Return a valid mocked structured output."""
    return {
        "checklist": {
            "maker": {"observation": "maker"},
            "profile": {"observation": "profile"},
            "year": {"observation": "year"},
        },
        "selection": selection,
    }


class V7PredictorTests(unittest.TestCase):
    def test_output_selection_is_limited_to_opaque_elements(self) -> None:
        output_model = models.nearest_output_model(["element_1", "element_2"])
        with self.assertRaises(ValidationError):
            output_model.model_validate(selected_payload("element_3"))

    def test_output_allows_not_found(self) -> None:
        output_model = models.nearest_output_model(["element_1", "element_2"])
        validated = output_model.model_validate(selected_payload(models.NOT_FOUND))
        self.assertEqual(validated.selection, models.NOT_FOUND)

    def test_system_message_contains_exact_schema_without_slugs(self) -> None:
        output_model = models.nearest_output_model(["element_1", "element_2"])
        output_format = models.response_format("ndr_select_nearest", output_model)
        system_prompt = predictor.system_prompt_with_schema("SELECT", output_format)

        self.assertIn('"enum": [', system_prompt)
        self.assertIn('"element_1"', system_prompt)
        self.assertIn('"not_found"', system_prompt)
        self.assertIn('"additionalProperties": false', system_prompt)
        self.assertNotIn('"slug"', system_prompt)

    def test_selection_content_contains_images_but_no_catalog_text(self) -> None:
        png_signature = b"\x89PNG\r\n\x1a\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            query = root / "query.png"
            first = root / "first.png"
            second = root / "second.png"
            for path in (query, first, second):
                path.write_bytes(png_signature)

            candidates = [
                {
                    "slug": "secret-first-slug",
                    "name": "Secret First Name",
                    "vintage": "2019",
                    "reference_image_path": str(first),
                },
                {
                    "slug": "secret-second-slug",
                    "name": "Secret Second Name",
                    "vintage": "2020",
                    "reference_image_path": str(second),
                },
            ]
            api_content, _ = predictor.selection_content(
                str(query),
                candidates,
                "original",
            )

        text_content = "\n".join(
            block["text"] for block in api_content if block["type"] == "text"
        )
        image_content = [block for block in api_content if block["type"] == "image_url"]
        self.assertEqual(len(image_content), 3)
        self.assertIn("element_1", text_content)
        self.assertIn("element_2", text_content)
        self.assertNotIn("CARD_JSON", text_content)
        self.assertNotIn("secret-first-slug", text_content)
        self.assertNotIn("Secret First Name", text_content)
        self.assertNotIn("2019", text_content)

    def test_predict_maps_opaque_element_to_slug_in_one_call(self) -> None:
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
                    "select_nearest": {"content": "SELECT BY IMAGE ONLY"},
                },
                "runtime": {
                    "model": "test/model",
                    "api_key_env": "V7_TEST_API_KEY",
                    "image_detail": "original",
                },
            }

            with (
                patch.dict(os.environ, {"V7_TEST_API_KEY": "secret"}),
                patch.object(
                    predictor,
                    "call_model",
                    return_value={
                        "status": "ok",
                        "error": None,
                        "selected": selected_payload("element_2"),
                        "trace": {},
                    },
                ) as call_model,
            ):
                result = predictor.predict(request)

        self.assertEqual(result["slug"], "second")
        self.assertEqual(result["_status"], "ok")
        self.assertEqual(
            result["_trace"]["pipeline"],
            "select_nearest_image_only_once",
        )
        self.assertEqual(
            result["_trace"]["element_mapping"],
            {"element_1": "first", "element_2": "second"},
        )
        self.assertEqual(call_model.call_count, 1)
        self.assertEqual(
            call_model.call_args.kwargs["system_prompt"],
            "SELECT BY IMAGE ONLY",
        )
        output_schema = call_model.call_args.kwargs["output_format"]
        selection_schema = output_schema["json_schema"]["schema"]["properties"][
            "selection"
        ]
        self.assertEqual(
            selection_schema["enum"],
            ["element_1", "element_2", models.NOT_FOUND],
        )


if __name__ == "__main__":
    unittest.main()
