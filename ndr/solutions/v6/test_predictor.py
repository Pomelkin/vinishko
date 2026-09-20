"""No-network contract and pipeline tests for the v6 pairwise predictor."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from ndr.solutions.v6 import models
from ndr.solutions.v6 import predictor


def evidence_payload(**states: str) -> dict:
    """Build a complete eight-discriminator payload for focused tests."""
    checklist = {}
    for name in models.DISCRIMINATORS:
        state = states.get(name, "unknown")
        checklist[name] = {
            "query_value": f"query {name}",
            "element_value": f"element {name}",
            "state": state,
        }
    return {"checklist": checklist}


class V6ModelTests(unittest.TestCase):
    def test_schema_requires_exactly_eight_three_state_discriminators(self) -> None:
        output_model = models.comparison_output_model(year_matters=True)
        schema = output_model.model_json_schema(mode="validation")
        checklist_schema = schema["$defs"]["ComparisonChecklist"]
        evidence_schema = schema["$defs"]["EvidenceItem"]

        self.assertEqual(
            tuple(checklist_schema["properties"]),
            models.DISCRIMINATORS,
        )
        self.assertEqual(
            set(checklist_schema["required"]),
            set(models.DISCRIMINATORS),
        )
        self.assertFalse(checklist_schema["additionalProperties"])
        self.assertEqual(
            evidence_schema["properties"]["state"]["enum"],
            ["match", "conflict", "unknown"],
        )
        self.assertFalse(evidence_schema["additionalProperties"])
        self.assertNotIn("verdict", schema["properties"])

    def test_contract_rejects_old_boolean_state_and_extra_fields(self) -> None:
        payload = evidence_payload()
        payload["checklist"]["maker"]["state"] = "same"
        payload["checklist"]["maker"]["is_same"] = True

        with self.assertRaises(ValidationError):
            models.YearNotMatterOutput.model_validate(payload)

    def test_unknown_maker_does_not_block_strong_product_identity(self) -> None:
        payload = evidence_payload(
            product_name="match",
            line_variant="match",
        )
        validated = models.YearNotMatterOutput.model_validate(payload).model_dump()

        self.assertEqual(
            models.comparison_verdict(validated, year_matters=False),
            "same",
        )

    def test_active_conflict_is_a_hard_reject(self) -> None:
        payload = evidence_payload(
            maker="match",
            product_name="match",
            sweetness="conflict",
        )
        validated = models.YearNotMatterOutput.model_validate(payload).model_dump()

        self.assertEqual(
            models.comparison_verdict(validated, year_matters=False),
            "different",
        )

    def test_required_vintage_must_match_exactly(self) -> None:
        base = {
            "maker": "match",
            "product_name": "match",
        }
        unknown = models.YearMattersOutput.model_validate(
            evidence_payload(**base, vintage="unknown"),
        ).model_dump()
        matching = models.YearMattersOutput.model_validate(
            evidence_payload(**base, vintage="match"),
        ).model_dump()
        conflicting = models.YearMattersOutput.model_validate(
            evidence_payload(**base, vintage="conflict"),
        ).model_dump()

        self.assertEqual(
            models.comparison_verdict(unknown, year_matters=True),
            "different",
        )
        self.assertEqual(
            models.comparison_verdict(matching, year_matters=True),
            "same",
        )
        self.assertEqual(
            models.comparison_verdict(conflicting, year_matters=True),
            "different",
        )

    def test_non_vintage_branch_records_but_ignores_vintage_conflict(self) -> None:
        payload = evidence_payload(
            maker="match",
            product_name="match",
            vintage="conflict",
        )
        validated = models.YearNotMatterOutput.model_validate(payload).model_dump()

        self.assertEqual(
            models.comparison_verdict(validated, year_matters=False),
            "same",
        )

    def test_unknown_without_identity_anchor_is_not_enough(self) -> None:
        payload = evidence_payload(maker="match")
        validated = models.YearNotMatterOutput.model_validate(payload).model_dump()

        self.assertEqual(
            models.comparison_verdict(validated, year_matters=False),
            "different",
        )

    def test_maker_grape_and_style_are_a_fallback_identity(self) -> None:
        payload = evidence_payload(
            maker="match",
            grape="match",
            color="match",
        )
        validated = models.YearNotMatterOutput.model_validate(payload).model_dump()

        self.assertEqual(
            models.comparison_verdict(validated, year_matters=False),
            "same",
        )


class V6PredictorSmokeTests(unittest.TestCase):
    def test_model_calls_never_retry(self) -> None:
        self.assertEqual(predictor.MAX_429_RETRIES, 0)

    def test_pairwise_pipeline_uses_deterministic_verdict_without_network(self) -> None:
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
                            "vintage": "",
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
                    "compare_year_matters": {"content": "EXACT YEAR"},
                    "compare_year_not_matter": {"content": "IGNORE YEAR"},
                    "resolve_multiple_same": {"content": "RESOLVE"},
                },
                "runtime": {
                    "model": "test/model",
                    "api_key_env": "V6_TEST_API_KEY",
                    "image_detail": "original",
                },
            }
            first_evidence = models.YearNotMatterOutput.model_validate(
                evidence_payload(
                    product_name="match",
                    line_variant="match",
                ),
            ).model_dump(mode="json")
            second_evidence = models.YearNotMatterOutput.model_validate(
                evidence_payload(
                    product_name="match",
                    line_variant="conflict",
                ),
            ).model_dump(mode="json")

            with (
                patch.dict(os.environ, {"V6_TEST_API_KEY": "secret"}),
                patch.object(
                    predictor,
                    "call_model",
                    side_effect=[
                        {
                            "status": "ok",
                            "error": None,
                            "selected": first_evidence,
                            "trace": {},
                        },
                        {
                            "status": "ok",
                            "error": None,
                            "selected": second_evidence,
                            "trace": {},
                        },
                    ],
                ) as call_model,
            ):
                result = predictor.predict(request)

        self.assertEqual(result["slug"], "first")
        self.assertEqual(result["_status"], "ok")
        self.assertEqual(call_model.call_count, 2)
        self.assertEqual(
            [call["verdict"] for call in result["_trace"]["comparison_calls"]],
            ["same", "different"],
        )
        self.assertIsNone(result["_trace"]["resolution_call"])


if __name__ == "__main__":
    unittest.main()
