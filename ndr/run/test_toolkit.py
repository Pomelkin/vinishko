"""Regression tests for the generated near-duplicate runner data."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATASET = REPO / "ndr" / "dataset"
sys.path.insert(0, str(HERE))

import build_dataset  # noqa: E402
import run as runner  # noqa: E402
from contracts import ContractError  # noqa: E402
from contracts import NOT_FOUND  # noqa: E402
from contracts import prediction_slug  # noqa: E402
from ndr.solutions.baseline import config as solution_config  # noqa: E402
from ndr.solutions.baseline import models as solution_models  # noqa: E402
from ndr.solutions.baseline import predictor as solution_predictor  # noqa: E402
from ndr.solutions.v1 import predictor as v1_predictor  # noqa: E402
from ndr.solutions.v2 import predictor as v2_predictor  # noqa: E402
from ndr.solutions.v3 import predictor as v3_predictor  # noqa: E402
from ndr.solutions.v4 import predictor as v4_predictor  # noqa: E402
from ndr.solutions.v6 import predictor as v6_predictor  # noqa: E402
from ndr.solutions.v7 import predictor as v7_predictor  # noqa: E402
try:
    from ndr.solutions.v5 import predictor as v5_predictor  # noqa: E402
except ModuleNotFoundError as error:  # v5 is optional until its experiment exists.
    if error.name != "ndr.solutions.v5":
        raise
    v5_predictor = None


RETRY_PREDICTORS = (
    ("baseline", solution_predictor),
    ("v1", v1_predictor),
    ("v2", v2_predictor),
    ("v3", v3_predictor),
)
SINGLE_ATTEMPT_PREDICTORS = (
    ("v4", v4_predictor),
    ("v6", v6_predictor),
    ("v7", v7_predictor),
)
if v5_predictor is not None:
    SINGLE_ATTEMPT_PREDICTORS += (("v5", v5_predictor),)
ALL_PREDICTORS = RETRY_PREDICTORS + SINGLE_ATTEMPT_PREDICTORS


def load_jsonl(name: str) -> list[dict]:
    """Load a generated dataset file."""
    with (DATASET / name).open(encoding="utf-8") as stream:
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
        build_dataset.check_documents(DATASET, documents)

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
            all(
                row["reason"]
                in {
                    "no_confirmed_near_duplicates",
                    "no_confirmed_near_duplicate_mapping",
                }
                for row in self.excluded
            ),
        )

    def test_gold_and_references_are_valid(self) -> None:
        groups = {row["group_id"]: row for row in self.groups}
        catalog = {row["slug"]: row for row in self.catalog}
        for case in self.manifest:
            self.assertTrue(
                case["expected_slug"] == NOT_FOUND
                or case["expected_slug"] in groups[case["group_id"]]["slugs"],
            )
            self.assertTrue((REPO / case["image_path"]).is_file())
        for group in self.groups:
            for slug in group["slugs"]:
                self.assertIn(slug, catalog)
        for card in self.catalog:
            self.assertTrue((REPO / card["reference_image_path"]).is_file())

    def test_not_found_registry_maps_only_to_catalog_slugs(self) -> None:
        rows = build_dataset.read_csv(
            build_dataset.NOT_FOUND_REGISTRY_PATH,
            build_dataset.REGISTRY_FIELDS,
        )
        catalog_slugs = {
            row["Slug"] for row in build_dataset.read_csv(build_dataset.CATALOG_PATH)
        }
        mapped_ids = set()
        for row in rows:
            self.assertTrue(row["slug_1"].startswith(build_dataset.NOT_FOUND_PREFIX))
            self.assertNotIn(row["slug_1"], catalog_slugs)
            self.assertIn(row["slug_2"], catalog_slugs)
            self.assertNotIn(row["slug_1"], mapped_ids)
            mapped_ids.add(row["slug_1"])

    def test_mapped_not_found_query_uses_full_catalog_component(self) -> None:
        not_found_id = (
            "not_found_fanagoriya-primum-alveus-blanc-de-blancs-bryut-2019"
        )
        anchor = (
            "fanagoriya-primum-alveus-blanc-de-blancs-ekstra-bryut-2019-"
            "shardone-beloe-11-13"
        )
        cases = [case for case in self.manifest if case.get("not_found_id") == not_found_id]
        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case["expected_slug"], NOT_FOUND)
        self.assertEqual(case["near_duplicate_anchor_slug"], anchor)

        registry_rows = build_dataset.read_csv(
            build_dataset.REGISTRY_PATH,
            build_dataset.REGISTRY_FIELDS,
        )
        adjacency = {}
        for row in registry_rows:
            adjacency.setdefault(row["slug_1"], set()).add(row["slug_2"])
            adjacency.setdefault(row["slug_2"], set()).add(row["slug_1"])
        expected_component = set(
            build_dataset.graph_components(adjacency, [anchor])[0],
        )
        group = next(row for row in self.groups if row["group_id"] == case["group_id"])
        self.assertEqual(set(group["slugs"]), expected_component)

    def test_unmapped_not_found_queries_are_explicitly_excluded(self) -> None:
        mapped_ids = {
            row["slug_1"]
            for row in build_dataset.read_csv(
                build_dataset.NOT_FOUND_REGISTRY_PATH,
                build_dataset.REGISTRY_FIELDS,
            )
        }
        source_ids = {
            path.name
            for path in build_dataset.TEST_PATH.iterdir()
            if path.is_dir() and path.name.startswith(build_dataset.NOT_FOUND_PREFIX)
        }
        excluded_ids = {
            row["not_found_id"]
            for row in self.excluded
            if row["reason"] == "no_confirmed_near_duplicate_mapping"
        }
        self.assertEqual(excluded_ids, source_ids - mapped_ids)
        self.assertTrue(
            all(row["expected_slug"] == NOT_FOUND for row in self.excluded if row.get("not_found_id")),
        )

    def test_not_found_ids_never_become_catalog_candidates(self) -> None:
        candidate_slugs = {row["slug"] for row in self.catalog}
        group_slugs = {slug for group in self.groups for slug in group["slugs"]}
        self.assertFalse(
            any(slug.startswith(build_dataset.NOT_FOUND_PREFIX) for slug in candidate_slugs),
        )
        self.assertFalse(
            any(slug.startswith(build_dataset.NOT_FOUND_PREFIX) for slug in group_slugs),
        )

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

    def test_predictor_matrix_covers_every_solution(self) -> None:
        solution_names = {
            path.name
            for path in (REPO / "ndr" / "solutions").iterdir()
            if path.is_dir() and (path / "predictor.py").is_file()
        }
        self.assertEqual(
            {solution_name for solution_name, _ in ALL_PREDICTORS},
            solution_names,
        )

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

    def test_429_retries_with_exponential_full_jitter(self) -> None:
        for solution_name, predictor in RETRY_PREDICTORS:
            with self.subTest(solution=solution_name):
                rate_limited = [
                    urllib.error.HTTPError(
                        "https://example.invalid/v1/chat/completions",
                        429,
                        "Too Many Requests",
                        None,
                        io.BytesIO(b'{"error":"rate limited"}'),
                    )
                    for _ in range(2)
                ]
                response = MagicMock()
                response.__enter__.return_value = response
                response.status = 200
                response.headers = {}
                response.read.return_value = (
                    b'{"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}]}'
                )
                with (
                    patch.object(
                        predictor.urllib.request,
                        "urlopen",
                        side_effect=[*rate_limited, response],
                    ) as urlopen,
                    patch.object(
                        predictor.random,
                        "uniform",
                        side_effect=[0.25, 1.5],
                    ) as jitter,
                    patch.object(predictor.time, "sleep") as sleep,
                ):
                    result = predictor.call_model(
                        runtime={
                            "api_base": "https://example.invalid/v1",
                            "model": "test/model",
                            "generation": {},
                            "provider": {},
                            "timeout": 60,
                        },
                        api_key="secret",
                        system_prompt="system",
                        api_content=[],
                        trace_content=[],
                        output_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "test",
                                "schema": {"type": "object"},
                            },
                        },
                        validator=lambda payload: payload,
                    )

                self.assertEqual(result["status"], "ok")
                self.assertEqual(urlopen.call_count, 3)
                self.assertEqual(
                    [item.args for item in jitter.call_args_list],
                    [(0.0, 1.0), (0.0, 2.0)],
                )
                self.assertEqual(
                    [item.args for item in sleep.call_args_list],
                    [(0.25,), (1.5,)],
                )
                self.assertEqual(len(result["trace"]["retry_attempts"]), 2)

    def test_single_attempt_predictors_do_not_retry_429(self) -> None:
        for solution_name, predictor in SINGLE_ATTEMPT_PREDICTORS:
            with self.subTest(solution=solution_name):
                rate_limited = urllib.error.HTTPError(
                    "https://example.invalid/v1/chat/completions",
                    429,
                    "Too Many Requests",
                    None,
                    io.BytesIO(b'{"error":"rate limited"}'),
                )
                with (
                    patch.object(
                        predictor.urllib.request,
                        "urlopen",
                        side_effect=rate_limited,
                    ) as urlopen,
                    patch.object(predictor.random, "uniform") as jitter,
                    patch.object(predictor.time, "sleep") as sleep,
                ):
                    result = predictor.call_model(
                        runtime={
                            "api_base": "https://example.invalid/v1",
                            "model": "test/model",
                            "generation": {},
                            "provider": {},
                            "timeout": 60,
                        },
                        api_key="secret",
                        system_prompt="system",
                        api_content=[],
                        trace_content=[],
                        output_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "test",
                                "schema": {"type": "object"},
                            },
                        },
                        validator=lambda payload: payload,
                    )

                self.assertEqual(result["status"], "predictor_error")
                self.assertEqual(urlopen.call_count, 1)
                jitter.assert_not_called()
                sleep.assert_not_called()
                self.assertEqual(result["trace"]["retry_attempts"], [])

    def test_only_429_is_retried_and_retry_count_is_capped_at_five(self) -> None:
        def http_error(status: int, reason: str) -> urllib.error.HTTPError:
            return urllib.error.HTTPError(
                "https://example.invalid/v1/chat/completions",
                status,
                reason,
                None,
                io.BytesIO(b'{"error":"failed"}'),
            )

        arguments = {
            "runtime": {
                "api_base": "https://example.invalid/v1",
                "model": "test/model",
                "generation": {},
                "provider": {},
                "timeout": 60,
            },
            "api_key": "secret",
            "system_prompt": "system",
            "api_content": [],
            "trace_content": [],
            "output_format": {
                "type": "json_schema",
                "json_schema": {"name": "test", "schema": {"type": "object"}},
            },
            "validator": lambda payload: payload,
        }
        for solution_name, predictor in ALL_PREDICTORS:
            with self.subTest(solution=solution_name, status=503):
                with (
                    patch.object(
                        predictor.urllib.request,
                        "urlopen",
                        side_effect=http_error(503, "Service Unavailable"),
                    ) as urlopen,
                    patch.object(predictor.random, "uniform") as jitter,
                    patch.object(predictor.time, "sleep") as sleep,
                ):
                    unavailable = predictor.call_model(**arguments)

                self.assertEqual(unavailable["status"], "predictor_error")
                self.assertEqual(urlopen.call_count, 1)
                jitter.assert_not_called()
                sleep.assert_not_called()

            with self.subTest(solution=solution_name, status=429):
                with (
                    patch.object(
                        predictor.urllib.request,
                        "urlopen",
                        side_effect=[
                            http_error(429, "Too Many Requests")
                            for _ in range(predictor.MAX_429_RETRIES + 1)
                        ],
                    ) as urlopen,
                    patch.object(predictor.random, "uniform", return_value=0.0),
                    patch.object(predictor.time, "sleep") as sleep,
                ):
                    exhausted = predictor.call_model(**arguments)

                self.assertEqual(exhausted["status"], "predictor_error")
                self.assertEqual(exhausted["trace"]["response_status"], 429)
                self.assertEqual(urlopen.call_count, predictor.MAX_429_RETRIES + 1)
                self.assertEqual(sleep.call_count, predictor.MAX_429_RETRIES)
                self.assertEqual(
                    len(exhausted["trace"]["retry_attempts"]),
                    predictor.MAX_429_RETRIES,
                )

    def test_embedded_provider_error_is_not_a_contract_error(self) -> None:
        body = (
            b'{"id":"generation-id","error":{"message":"A Timeout Occurred",'
            b'"code":504,"metadata":{"error_type":"timeout"}}}'
        )
        for solution_name, predictor in ALL_PREDICTORS:
            with self.subTest(solution=solution_name):
                response = MagicMock()
                response.__enter__.return_value = response
                response.status = 200
                response.headers = {}
                response.read.return_value = body
                with patch.object(
                    predictor.urllib.request,
                    "urlopen",
                    return_value=response,
                ) as urlopen:
                    result = predictor.call_model(
                        runtime={
                            "api_base": "https://example.invalid/v1",
                            "model": "test/model",
                            "generation": {},
                            "provider": {},
                            "timeout": 1,
                        },
                        api_key="secret",
                        system_prompt="system",
                        api_content=[],
                        trace_content=[],
                        output_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "test",
                                "schema": {"type": "object"},
                            },
                        },
                        validator=lambda payload: payload,
                    )

                self.assertEqual(urlopen.call_count, 1)
                self.assertEqual(result["status"], "predictor_error")
                self.assertEqual(
                    result["error"],
                    "OpenRouter error 504: A Timeout Occurred",
                )
                self.assertEqual(result["trace"]["response_status"], 200)
                self.assertEqual(result["trace"]["provider_error"]["code"], 504)
                self.assertEqual(result["trace"]["generations"], [])
                self.assertEqual(result["trace"]["retry_attempts"], [])

    def test_response_heartbeats_cannot_extend_wall_clock_deadline(self) -> None:
        for solution_name, predictor in ALL_PREDICTORS:
            with self.subTest(solution=solution_name):
                response = MagicMock()
                response.__enter__.return_value = response
                response.status = 200
                response.headers = {}
                response.read1.return_value = b" \n"
                with (
                    patch.object(
                        predictor.urllib.request,
                        "urlopen",
                        return_value=response,
                    ) as urlopen,
                    patch.object(
                        predictor.time,
                        "monotonic",
                        side_effect=[0.0, 0.0, 0.25, 0.5, 1.01],
                    ),
                ):
                    result = predictor.call_model(
                        runtime={
                            "api_base": "https://example.invalid/v1",
                            "model": "test/model",
                            "generation": {},
                            "provider": {},
                            "timeout": 1,
                        },
                        api_key="secret",
                        system_prompt="system",
                        api_content=[],
                        trace_content=[],
                        output_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "test",
                                "schema": {"type": "object"},
                            },
                        },
                        validator=lambda payload: payload,
                    )

                self.assertEqual(urlopen.call_count, 1)
                self.assertEqual(result["status"], "predictor_error")
                self.assertEqual(
                    result["error"],
                    "TimeoutError: model call exceeded 1s wall-clock deadline",
                )
                self.assertEqual(result["trace"]["response_status"], 200)
                self.assertEqual(result["trace"]["raw_response_text"], " \n")
                self.assertEqual(
                    result["trace"]["timeout"],
                    {"kind": "wall_clock", "limit_seconds": 1.0},
                )


class RunnerTests(unittest.TestCase):
    """Protect dotenv loading, aggregate metrics, and result artifacts."""

    def test_solution_can_declare_one_named_prompt(self) -> None:
        solution = REPO / "ndr" / "solutions" / "v7"
        self.assertEqual(
            runner.solution_prompt_paths(solution),
            {"select_nearest": solution / "prompts" / "select_nearest.txt"},
        )

    def test_copied_solution_is_discovered_selected_and_versioned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            solutions = Path(directory) / "solutions"
            copied = solutions / "эксперимент-2"
            shutil.copytree(REPO / "ndr" / "solutions" / "baseline", copied)

            self.assertEqual(runner.discover_solutions(solutions), ["эксперимент-2"])
            self.assertEqual(
                runner.solution_path(solutions, "эксперимент-2"),
                copied.resolve(),
            )
            first_fingerprint, first_files = runner.solution_manifest(copied)
            prompt = copied / "prompts" / "compare_year_matters.txt"
            prompt.write_text(
                f"{prompt.read_text(encoding='utf-8')}\nexperiment marker\n",
                encoding="utf-8",
            )
            second_fingerprint, second_files = runner.solution_manifest(copied)

            self.assertNotEqual(first_fingerprint, second_fingerprint)
            self.assertNotEqual(
                first_files["prompts/compare_year_matters.txt"],
                second_files["prompts/compare_year_matters.txt"],
            )

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
                "expected_slug": "a",
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
                "expected_slug": "a",
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
            rows,
            run_id="test",
            run_timestamp="20260919T000000.000000Z",
            solution_name="baseline",
            solution_fingerprint="abc123",
            model="test/model",
            elapsed_seconds=2,
        )
        self.assertEqual(metrics["quality"]["attempted_queries"], 2)
        self.assertEqual(metrics["quality"]["queries"], 1)
        self.assertEqual(metrics["quality"]["excluded_errors"], 1)
        self.assertEqual(metrics["quality"]["accuracy"], 1.0)
        self.assertEqual(metrics["quality"]["answered"], 1)
        self.assertEqual(metrics["quality"]["not_found"], 0)
        self.assertEqual(metrics["quality"]["not_found_metrics"]["support"], 0)
        self.assertIsNone(metrics["quality"]["not_found_metrics"]["precision"])
        self.assertIsNone(metrics["quality"]["not_found_metrics"]["recall"])
        self.assertEqual(metrics["quality"]["not_found_metrics"]["accuracy"], 1.0)
        self.assertEqual(metrics["quality"]["predictor_errors"], 1)
        self.assertNotIn("accuracy_when_answered", metrics["quality"])
        self.assertNotIn("coverage", metrics["quality"])
        self.assertEqual(set(metrics["by_group_size"]), {"2", "3"})
        self.assertEqual(metrics["model_activity"]["requests"], 3)
        self.assertEqual(metrics["model_activity"]["successful_requests"], 2)
        self.assertEqual(metrics["model_activity"]["failed_requests"], 1)
        self.assertEqual(metrics["model_activity"]["generations_returned"], 3)
        self.assertEqual(metrics["tie_breaker"]["queries"], 1)
        self.assertEqual(metrics["tie_breaker"]["share_of_attempted_queries"], 0.5)
        self.assertEqual(metrics["model_activity"]["usage"]["prompt_tokens"], 12)
        self.assertEqual(
            metrics["model_activity"]["usage"]["completion_tokens_details.reasoning_tokens"],
            2,
        )
        self.assertNotIn("ignored_boolean", metrics["model_activity"]["usage"])

    def test_metrics_omit_tie_breaker_for_predictor_without_one(self) -> None:
        rows = [{
            "expected_slug": "a",
            "input": {"candidate_order": ["a"]},
            "prediction": {
                "slug": "a", "correct": True, "status": "ok", "latency_ms": 1,
            },
            "predictor_response": {"slug": "a"},
        }]

        metrics = runner.build_metrics(
            rows,
            run_id="test",
            run_timestamp="20260919T000000.000000Z",
            solution_name="without-tie-breaker",
            solution_fingerprint="abc123",
            model=None,
            elapsed_seconds=1,
        )

        self.assertNotIn("tie_breaker", metrics)

    def test_tie_breaker_share_counts_a_failed_call(self) -> None:
        rows = [
            {
                "predictor_response": {"_trace": {
                    "resolution_call": {"status": "predictor_error"},
                }},
            },
            {
                "predictor_response": {"_trace": {"resolution_call": None}},
            },
        ]

        activity = runner.tie_breaker_metrics(rows)

        self.assertIsNotNone(activity)
        self.assertEqual(activity["queries"], 1)
        self.assertEqual(activity["share_of_attempted_queries"], 0.5)

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
                "expected_slug": "a",
                "prediction": {
                    "slug": "a",
                    "correct": True,
                    "status": "ok",
                    "latency_ms": 10,
                },
            },
            {
                "expected_slug": "a",
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
        self.assertEqual(quality["not_found_metrics"]["support"], 0)
        self.assertEqual(quality["not_found_metrics"]["predicted"], 1)
        self.assertEqual(quality["not_found_metrics"]["false_positives"], 1)
        self.assertEqual(quality["not_found_metrics"]["true_negatives"], 1)
        self.assertEqual(quality["not_found_metrics"]["precision"], 0.0)
        self.assertIsNone(quality["not_found_metrics"]["recall"])
        self.assertEqual(quality["not_found_metrics"]["f1"], 0.0)
        self.assertEqual(quality["not_found_metrics"]["accuracy"], 0.5)

    def test_not_found_metrics_treat_not_found_as_binary_class(self) -> None:
        rows = [
            {
                "expected_slug": NOT_FOUND,
                "prediction": {
                    "slug": NOT_FOUND, "correct": True,
                    "status": "ok", "latency_ms": 10,
                },
            },
            {
                "expected_slug": "catalog-a",
                "prediction": {
                    "slug": NOT_FOUND, "correct": False,
                    "status": "ok", "latency_ms": 10,
                },
            },
            {
                "expected_slug": NOT_FOUND,
                "prediction": {
                    "slug": "catalog-a", "correct": False,
                    "status": "ok", "latency_ms": 10,
                },
            },
            {
                "expected_slug": "catalog-a",
                "prediction": {
                    "slug": "catalog-a", "correct": True,
                    "status": "ok", "latency_ms": 10,
                },
            },
            {
                "expected_slug": NOT_FOUND,
                "prediction": {
                    "slug": None, "correct": None,
                    "status": "predictor_error", "latency_ms": 10,
                },
            },
        ]

        quality = runner.quality_metrics(rows)
        not_found_metrics = quality["not_found_metrics"]
        self.assertEqual(not_found_metrics["support"], 2)
        self.assertEqual(not_found_metrics["predicted"], 2)
        self.assertEqual(not_found_metrics["true_positives"], 1)
        self.assertEqual(not_found_metrics["false_positives"], 1)
        self.assertEqual(not_found_metrics["false_negatives"], 1)
        self.assertEqual(not_found_metrics["true_negatives"], 1)
        self.assertEqual(not_found_metrics["precision"], 0.5)
        self.assertEqual(not_found_metrics["recall"], 0.5)
        self.assertEqual(not_found_metrics["f1"], 0.5)
        self.assertEqual(not_found_metrics["accuracy"], 0.5)

    def test_smoke_runs_are_versioned_and_never_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results"
            solutions = Path(directory) / "solutions"
            shutil.copytree(
                REPO / "ndr" / "solutions" / "baseline",
                solutions / "experiment-v2",
            )
            argv = [
                "run.py", "--solution", "experiment-v2",
                "--solutions-dir", str(solutions),
                "--predictor", str(HERE / "example_predictor.py"),
                "--limit", "2", "--concurrency", "2",
                "--results-dir", str(results),
                "--env-file", str(Path(directory) / "missing.env"),
            ]
            with patch.object(sys, "argv", argv):
                self.assertEqual(runner.main(), 0)
            with patch.object(sys, "argv", argv):
                self.assertEqual(runner.main(), 0)

            runs = sorted((results / "experiment-v2").iterdir())
            self.assertEqual(len(runs), 2)
            first = runs[0]
            self.assertTrue((first / "run.json").is_file())
            self.assertTrue((first / "metrics.json").is_file())
            self.assertFalse((first / "solution").exists())
            self.assertEqual(len(list((first / "by_case").glob("*.json"))), 2)
            self.assertEqual(
                json.loads((first / "metrics.json").read_text(encoding="utf-8"))["quality"]["queries"],
                2,
            )
            run = json.loads((first / "run.json").read_text(encoding="utf-8"))["run"]
            self.assertEqual(run["solution"]["name"], "experiment-v2")
            self.assertEqual(len(run["solution"]["fingerprint"]), 64)


if __name__ == "__main__":
    unittest.main()
