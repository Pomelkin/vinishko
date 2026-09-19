"""Regression tests for the generated near-duplicate runner data."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import build_dataset  # noqa: E402
import run as runner  # noqa: E402
from contracts import ContractError  # noqa: E402
from contracts import NOT_FOUND  # noqa: E402
from contracts import prediction_slug  # noqa: E402
from ndr.solution import config as solution_config  # noqa: E402
from ndr.solution import models as solution_models  # noqa: E402
from ndr.solution import predictor as solution_predictor  # noqa: E402


def load_jsonl(name: str) -> list[dict]:
    """Load a generated dataset file."""
    with (HERE / "dataset" / name).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class DatasetTests(unittest.TestCase):
    """Protect selection, grouping, paths, and output contract invariants."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_jsonl("manifest.jsonl")
        cls.groups = load_jsonl("groups.jsonl")
        cls.catalog = load_jsonl("catalog.jsonl")
        cls.excluded = load_jsonl("excluded.jsonl")

    def test_generated_dataset_is_current(self) -> None:
        documents, _ = build_dataset.rendered_dataset()
        build_dataset.check_documents(HERE / "dataset", documents)

    def test_every_source_query_is_included_or_explicitly_excluded(self) -> None:
        all_paths = {
            path.relative_to(REPO).as_posix()
            for path in (REPO / "data" / "test").rglob("*")
            if path.is_file()
        }
        selected = {row["image_path"] for row in self.manifest}
        excluded = {row["image_path"] for row in self.excluded}
        self.assertEqual(all_paths, selected | excluded)
        self.assertFalse(selected & excluded)

    def test_no_singleton_group_is_tested(self) -> None:
        groups = {row["group_id"]: row for row in self.groups}
        self.assertTrue(all(len(row["slugs"]) >= 2 for row in self.groups))
        self.assertTrue(all(len(groups[row["group_id"]]["slugs"]) >= 2 for row in self.manifest))
        self.assertTrue(
            all(row["reason"] == "no_confirmed_near_duplicates" for row in self.excluded),
        )

    def test_gold_and_references_are_valid(self) -> None:
        groups = {row["group_id"]: row for row in self.groups}
        catalog = {row["slug"]: row for row in self.catalog}
        for case in self.manifest:
            self.assertIn(case["expected_slug"], groups[case["group_id"]]["slugs"])
            self.assertTrue((REPO / case["image_path"]).is_file())
        for group in self.groups:
            for slug in group["slugs"]:
                self.assertIn(slug, catalog)
        for card in self.catalog:
            self.assertTrue((REPO / card["reference_image_path"]).is_file())

    def test_prediction_contract(self) -> None:
        allowed = {"a", "b"}
        self.assertEqual(prediction_slug({"slug": "a"}, allowed), "a")
        self.assertEqual(prediction_slug({"slug": "not found"}, allowed), NOT_FOUND)
        self.assertEqual(prediction_slug({"slug": NOT_FOUND}, allowed), NOT_FOUND)
        with self.assertRaises(ContractError):
            prediction_slug({"slug": "outside"}, allowed)
        with self.assertRaises(ContractError):
            prediction_slug("a", allowed)


class SolutionIntegrationTests(unittest.TestCase):
    """Protect raw generation retention and the integrated response envelope."""

    def test_image_type_is_detected_from_signature(self) -> None:
        self.assertEqual(solution_predictor.image_media_type(b"\xff\xd8\xffx"), "image/jpeg")
        self.assertEqual(
            solution_predictor.image_media_type(b"RIFF\x00\x00\x00\x00WEBP"),
            "image/webp",
        )
        with self.assertRaises(ValueError):
            solution_predictor.image_media_type(b"not an image")

    def test_image_block_requests_and_traces_original_detail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "query.jpg"
            image_path.write_bytes(b"\xff\xd8\xfftest")
            api_block, trace_block = solution_predictor.image_blocks(
                str(image_path),
                "original",
            )

        self.assertEqual(api_block["image_url"]["detail"], "original")
        self.assertEqual(trace_block["image"]["detail"], "original")
        self.assertTrue(api_block["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_every_generation_and_reasoning_is_retained(self) -> None:
        raw = {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "content": '{"slug":"b"}',
                        "reasoning": "reasoning zero",
                        "reasoning_details": [{"type": "text", "text": "detail zero"}],
                    },
                },
                {
                    "index": 1,
                    "finish_reason": "stop",
                    "message": {
                        "content": '{"slug":"a"}',
                        "reasoning": "reasoning one",
                        "reasoning_details": [{"type": "text", "text": "detail one"}],
                    },
                },
            ],
        }
        resolver_model = solution_models.resolver_output_model(["a", "b"])
        generations, selected_index, selected = solution_predictor.generation_records(
            raw,
            lambda payload: solution_models.validate_output(payload, resolver_model),
        )
        self.assertEqual(len(generations), 2)
        self.assertEqual(generations[0]["reasoning"], "reasoning zero")
        self.assertEqual(
            generations[1]["reasoning_details"][0]["text"],
            "detail one",
        )
        self.assertEqual(selected_index, 0)
        self.assertEqual(selected, {"slug": "b"})

    def test_pydantic_outputs_are_maximally_restricted(self) -> None:
        valid = {
            "checklist": {
                "maker": {"observation": "same maker", "is_same": True},
                "profile": {"observation": "same profile", "is_same": True},
                "year": {"observation": "2019", "is_same": True},
            },
            "verdict": "same",
        }
        self.assertEqual(
            solution_models.validate_output(valid, solution_models.YearMattersOutput),
            valid,
        )

        inconsistent = {**valid, "verdict": "different"}
        with self.assertRaises(solution_models.ModelContractError):
            solution_models.validate_output(
                inconsistent,
                solution_models.YearMattersOutput,
            )

        extra_field = {**valid, "confidence": 1}
        with self.assertRaises(solution_models.ModelContractError):
            solution_models.validate_output(
                extra_field,
                solution_models.YearMattersOutput,
            )

        resolver_model = solution_models.resolver_output_model(["a", "b"])
        resolver_schema = resolver_model.model_json_schema()
        self.assertEqual(resolver_schema["properties"]["slug"]["enum"], ["a", "b"])
        self.assertFalse(resolver_schema["additionalProperties"])
        with self.assertRaises(solution_models.ModelContractError):
            solution_models.validate_output({"slug": "outside"}, resolver_model)

    def test_central_config_builds_generation_and_provider_payloads(self) -> None:
        routing = solution_config.ProviderRoutingSettings(
            routing_mode="throughput",
            only=("provider-a", "provider-b"),
            allow_fallbacks=False,
            require_parameters=True,
            data_collection="deny",
        )
        self.assertEqual(
            routing.request_payload(),
            {
                "only": ["provider-a", "provider-b"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
                "sort": "throughput",
            },
        )

        generation = solution_config.GenerationSettings(
            temperature=0.25,
            top_p=0.9,
            max_completion_tokens=123,
            generations=2,
            reasoning_effort="low",
            reasoning_exclude=True,
        )
        payload = solution_predictor.build_request_payload(
            runtime={
                "model": "test/model",
                "generation": generation.request_payload(),
                "provider": routing.request_payload(),
            },
            system_prompt="system",
            api_content=[],
            output_format={"type": "json_schema"},
        )
        self.assertEqual(payload["temperature"], 0.25)
        self.assertEqual(payload["top_p"], 0.9)
        self.assertEqual(payload["max_completion_tokens"], 123)
        self.assertEqual(payload["n"], 2)
        self.assertEqual(payload["reasoning"], {"effort": "low", "exclude": True})
        self.assertEqual(payload["provider"]["sort"], "throughput")

        single_generation = solution_config.GenerationSettings(reasoning_effort="none")
        single_payload = single_generation.request_payload()
        self.assertNotIn("n", single_payload)
        self.assertNotIn("image_detail", single_payload)
        self.assertEqual(
            single_payload["reasoning"],
            {"effort": "none", "exclude": False},
        )
        self.assertEqual(single_payload["max_completion_tokens"], 16000)

        with self.assertRaises(ValueError):
            solution_config.ProviderRoutingSettings(
                only=("provider-a",),
                ignore=("provider-a",),
            )

    def test_numeric_reasoning_effort_accepts_only_integers_from_1_to_100(self) -> None:
        for effort in (1, 5, 42, 100):
            generation = solution_config.GenerationSettings(reasoning_effort=effort)
            self.assertEqual(generation.request_payload()["reasoning"]["effort"], effort)
            self.assertEqual(runner.parse_reasoning_effort(str(effort)), effort)

        for effort in (0, 101, -1, 1.5, True):
            with self.assertRaises(ValueError):
                solution_config.GenerationSettings(reasoning_effort=effort)

        for effort in ("0", "101", "1.5", "unknown"):
            with self.assertRaises(argparse.ArgumentTypeError):
                runner.parse_reasoning_effort(effort)

        self.assertEqual(runner.parse_reasoning_effort("high"), "high")

    def test_case_artifact_preserves_predictor_response(self) -> None:
        response = {
            "slug": "a",
            "_status": "ok",
            "_error": None,
            "_trace": {
                "raw_response": {"choices": [{"message": {"reasoning": "all of it"}}]},
                "generations": [{"reasoning": "all of it"}],
            },
        }

        record = runner.execute_case(
            run_id="test-run",
            case={
                "query_id": "q-test",
                "group_id": "g-test",
                "expected_slug": "a",
            },
            request={
                "group": {"candidates": [{"slug": "a"}, {"slug": "b"}]},
            },
            input_record={"query": {"image_path": "anonymous"}},
            predict=lambda request: response,
        )
        self.assertIs(record["predictor_response"], response)
        self.assertEqual(
            record["predictor_response"]["_trace"]["generations"][0]["reasoning"],
            "all of it",
        )
        self.assertTrue(record["prediction"]["correct"])

    def test_three_stage_pipeline_resolves_multiple_same(self) -> None:
        request = {
            "query": {"image_path": "query"},
            "group": {
                "candidates": [
                    {"slug": "a", "vintage": "2019", "reference_image_path": "a"},
                    {"slug": "b", "vintage": "", "reference_image_path": "b"},
                ],
            },
            "prompts": {
                "compare_year_matters": {"content": "year matters"},
                "compare_year_not_matter": {"content": "year ignored"},
                "resolve_multiple_same": {"content": "resolve"},
            },
            "runtime": {
                "model": "test/model",
                "api_key_env": "NDR_TEST_API_KEY",
                "api_base": "https://example.invalid/v1",
            },
        }
        same_with_year = {
            "checklist": {
                "maker": {"observation": "same", "is_same": True},
                "profile": {"observation": "same", "is_same": True},
                "year": {"observation": "2019", "is_same": True},
            },
            "verdict": "same",
        }
        same_without_year = {
            "checklist": {
                "maker": {"observation": "same", "is_same": True},
                "profile": {"observation": "same", "is_same": True},
            },
            "verdict": "same",
        }
        responses = [
            {"status": "ok", "error": None, "selected": same_with_year, "trace": {"id": 1}},
            {"status": "ok", "error": None, "selected": same_without_year, "trace": {"id": 2}},
            {"status": "ok", "error": None, "selected": {"slug": "b"}, "trace": {"id": 3}},
        ]
        with (
            patch.dict(os.environ, {"NDR_TEST_API_KEY": "secret"}),
            patch.object(solution_predictor, "comparison_content", return_value=([], [])),
            patch.object(solution_predictor, "resolution_content", return_value=([], [])),
            patch.object(solution_predictor, "call_model", side_effect=responses) as call,
        ):
            result = solution_predictor.predict(request)

        self.assertEqual(result["slug"], "b")
        self.assertEqual(result["_trace"]["same_slugs"], ["a", "b"])
        self.assertEqual(len(result["_trace"]["comparison_calls"]), 2)
        self.assertEqual(call.call_count, 3)
        self.assertEqual(call.call_args_list[0].kwargs["system_prompt"], "year matters")
        self.assertEqual(call.call_args_list[1].kwargs["system_prompt"], "year ignored")
        self.assertEqual(call.call_args_list[2].kwargs["system_prompt"], "resolve")


class RunnerTests(unittest.TestCase):
    """Protect dotenv loading, aggregate metrics, and result artifacts."""

    def test_atomic_write_retries_transient_windows_destination_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run.json"
            target.write_text("{}\n", encoding="utf-8")
            real_replace = os.replace
            attempts = 0

            def flaky_replace(source: Path, destination: Path) -> None:
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    error = PermissionError("destination is temporarily locked")
                    error.winerror = 5
                    raise error
                real_replace(source, destination)

            with (
                patch.object(runner.os, "replace", side_effect=flaky_replace),
                patch.object(runner.time, "sleep") as sleep,
            ):
                runner.atomic_write_json(target, {"status": "completed"})

            self.assertEqual(attempts, 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                {"status": "completed"},
            )
            self.assertEqual(list(target.parent.glob(".run.json.*.tmp")), [])

    def test_project_env_is_loaded_without_overriding_process_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "# comment\n"
                "export OPENROUTER_API_KEY='secret # literal'\n"
                "OPENROUTER_MODEL=dotenv/model # inline comment\n"
                'OPENROUTER_API_BASE="https://example.test/v1"\n',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"OPENROUTER_MODEL": "process/model"}, clear=True):
                metadata = runner.load_env_file(env_path)
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "secret # literal")
                self.assertEqual(os.environ["OPENROUTER_MODEL"], "process/model")
                self.assertEqual(os.environ["OPENROUTER_API_BASE"], "https://example.test/v1")
                self.assertNotIn("secret # literal", json.dumps(metadata))
                self.assertNotIn("OPENROUTER_MODEL", metadata["loaded_keys"])

    def test_metrics_include_usage_activity_and_group_breakdown(self) -> None:
        def call(status: str, generations: int, usage: dict) -> dict:
            return {
                "status": status,
                "trace": {
                    "generations": [{}] * generations,
                    "raw_response": {"usage": usage},
                },
            }

        rows = [
            {
                "input": {"candidate_order": ["a", "b"]},
                "prediction": {
                    "slug": "a", "correct": True, "status": "ok", "latency_ms": 10,
                },
                "predictor_response": {"_trace": {
                    "comparison_calls": [call("ok", 2, {
                        "prompt_tokens": 10,
                        "completion_tokens": 3,
                        "completion_tokens_details": {"reasoning_tokens": 2},
                        "cost": 0.1,
                        "ignored_boolean": True,
                    })],
                    "resolution_call": call("ok", 1, {"prompt_tokens": 2, "cost": 0.05}),
                }},
            },
            {
                "input": {"candidate_order": ["a", "b", "c"]},
                "prediction": {
                    "slug": None, "correct": None,
                    "status": "predictor_error", "latency_ms": 20,
                },
                "predictor_response": {"_trace": {
                    "comparison_calls": [call("predictor_error", 0, {})],
                    "resolution_call": None,
                }},
            },
        ]
        metrics = runner.build_metrics(
            rows, run_id="test", model="test/model", elapsed_seconds=2,
        )
        self.assertEqual(metrics["quality"]["attempted_queries"], 2)
        self.assertEqual(metrics["quality"]["queries"], 1)
        self.assertEqual(metrics["quality"]["excluded_errors"], 1)
        self.assertEqual(metrics["quality"]["accuracy"], 1.0)
        self.assertEqual(metrics["quality"]["answered"], 1)
        self.assertEqual(metrics["quality"]["not_found"], 0)
        self.assertEqual(metrics["quality"]["predictor_errors"], 1)
        self.assertNotIn("accuracy_when_answered", metrics["quality"])
        self.assertNotIn("coverage", metrics["quality"])
        self.assertEqual(set(metrics["by_group_size"]), {"2", "3"})
        self.assertEqual(metrics["model_activity"]["requests"], 3)
        self.assertEqual(metrics["model_activity"]["successful_requests"], 2)
        self.assertEqual(metrics["model_activity"]["failed_requests"], 1)
        self.assertEqual(metrics["model_activity"]["generations_returned"], 3)
        self.assertEqual(metrics["model_activity"]["usage"]["prompt_tokens"], 12)
        self.assertEqual(
            metrics["model_activity"]["usage"]["completion_tokens_details.reasoning_tokens"],
            2,
        )
        self.assertNotIn("ignored_boolean", metrics["model_activity"]["usage"])

    def test_predictor_error_has_no_verdict_and_is_excluded_from_quality(self) -> None:
        record = runner.execute_case(
            run_id="test",
            case={
                "query_id": "q-error",
                "group_id": "g-error",
                "expected_slug": "a",
            },
            request={"group": {"candidates": [{"slug": "a"}]}},
            input_record={"candidate_order": ["a"]},
            predict=lambda _request: {
                "slug": None,
                "_status": "predictor_error",
                "_error": "upstream failed",
                "_trace": {},
            },
        )

        self.assertEqual(record["prediction"]["status"], "predictor_error")
        self.assertIsNone(record["prediction"]["slug"])
        self.assertIsNone(record["prediction"]["correct"])

        quality = runner.quality_metrics([record])
        self.assertEqual(quality["attempted_queries"], 1)
        self.assertEqual(quality["queries"], 0)
        self.assertEqual(quality["excluded_errors"], 1)
        self.assertEqual(quality["predictor_errors"], 1)
        self.assertEqual(quality["not_found"], 0)
        self.assertIsNone(quality["accuracy"])
        self.assertNotIn("accuracy_when_answered", quality)
        self.assertNotIn("coverage", quality)

    def test_not_found_is_an_answer_and_stays_in_accuracy_denominator(self) -> None:
        rows = [
            {
                "prediction": {
                    "slug": "a",
                    "correct": True,
                    "status": "ok",
                    "latency_ms": 10,
                },
            },
            {
                "prediction": {
                    "slug": NOT_FOUND,
                    "correct": False,
                    "status": "ok",
                    "latency_ms": 20,
                },
            },
        ]

        quality = runner.quality_metrics(rows)
        self.assertEqual(quality["queries"], 2)
        self.assertEqual(quality["answered"], 2)
        self.assertEqual(quality["not_found"], 1)
        self.assertEqual(quality["correct"], 1)
        self.assertEqual(quality["accuracy"], 0.5)

    def test_smoke_run_writes_metrics_and_requires_force(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results"
            argv = [
                "run.py", "--predictor", str(HERE / "example_predictor.py"),
                "--limit", "2", "--concurrency", "2",
                "--results-dir", str(results),
                "--env-file", str(Path(directory) / "missing.env"),
            ]
            with patch.object(sys, "argv", [*argv, "--force"]):
                self.assertEqual(runner.main(), 0)
            self.assertTrue((results / "run.json").is_file())
            self.assertTrue((results / "metrics.json").is_file())
            self.assertEqual(len(list((results / "by_case").glob("*.json"))), 2)
            self.assertEqual(
                json.loads((results / "metrics.json").read_text(encoding="utf-8"))["quality"]["queries"],
                2,
            )
            with patch.object(sys, "argv", argv):
                self.assertEqual(runner.main(), 1)


if __name__ == "__main__":
    unittest.main()
