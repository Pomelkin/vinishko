"""Офлайн-проверки второго уровня: без qdrant, S3 и модели. Запуск из корня: python -m unittest vinishko.pipeline.steps.filter.test_resolve -v"""

import csv
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import ClassVar
from typing import cast
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
from PIL import Image

from vinishko.pipeline.catalog import Catalog
from vinishko.pipeline.pipeline import Pipeline
from vinishko.pipeline.pipeline import Searcher
from vinishko.pipeline.steps.filter import predictor
from vinishko.pipeline.steps.filter.configs import DEFAULT_CONFIG
from vinishko.pipeline.steps.filter.configs import (
    load_config as load_ndr_config,
)
from vinishko.pipeline.steps.filter.resolve import CSV_FIELDS
from vinishko.pipeline.steps.filter.resolve import NDR_CONCURRENCY
from vinishko.pipeline.steps.filter.resolve import FilterError
from vinishko.pipeline.steps.filter.resolve import FilterResolver
from vinishko.pipeline.steps.filter.resolve import candidate_card
from vinishko.pipeline.steps.filter.resolve import load_cards
from vinishko.pipeline.steps.normalization.normalize import Normalizer
from vinishko.pipeline.steps.vis_searcher.search import SearchReason
from vinishko.pipeline.steps.vis_searcher.search import VisSearcher
from vinishko.pipeline.steps.vis_searcher.configs import load_config as load_search_config
from vinishko.pipeline.steps.vis_searcher.configs import GroupSearch
from vinishko.pipeline.evaluate_pipeline import e2e_row, e2e_metrics, run
from vinishko.pipeline.steps.vis_searcher.evaluate import Row as SearchRow
from vinishko.pipeline.pipeline import PipelineResult
from vinishko.pipeline.structs import BottleCandidates
from vinishko.pipeline.structs import BottleCrop
from vinishko.pipeline.structs import Candidate
from vinishko.pipeline.structs import MatchedBottle
from vinishko.pipeline.structs import Rejection
from vinishko.pipeline.structs import UnmatchedBottle

PREDICT = "vinishko.pipeline.steps.filter.resolve.predict"


def crop(index: int = 1) -> BottleCrop:
    """Минимальная бутылка с валидными RGB-кропами."""
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    return BottleCrop(index, 0.9, [], [], 0.0, rgb, {}, rgb.copy(), {})


def candidate(
        bottle: BottleCrop, slug: str, group: str, members: list[str]
) -> Candidate:
    """Кандидат с метаданными точки, которые читает второй уровень."""
    return Candidate(
        slug=slug,
        score=0.8,
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        crop=bottle,
        group=group,
        retrieved=True,
        payload={
            "slug": slug,
            "source_image": f"{slug}.webp",
            "grape": "Пино нуар",
            "vintage": "2023",
        },
        scores={"crop": 0.8},
    )


def model_response(slug: str) -> dict:
    """Валидный ответ подменённого predictor.predict."""
    selected = {
        "slug": slug,
        "checklist": {
            "maker": {"observation": "совпадает"},
            "profile": {"observation": "совпадает"},
            "year": {"observation": "2023"},
        },
    }
    return {
        "slug": slug,
        "_status": "ok",
        "_error": None,
        "_trace": {"comparison_calls": [{"selected": selected}]},
    }


class NormalizerStub:
    """Нормализатор, отдающий заранее готовую бутылку."""

    crop_config: ClassVar[dict] = {}

    def __init__(self, bottle: BottleCrop) -> None:
        self.bottle = bottle

    def __call__(self, _image: str) -> list[BottleCrop]:
        return [self.bottle]


class SearcherStub:
    """Поиск, отдающий заранее готовые ответы."""

    def __init__(self, results: list[BottleCandidates | UnmatchedBottle]) -> None:
        self.results = results

    def __call__(
            self, _crops: list[BottleCrop]
    ) -> list[BottleCandidates | UnmatchedBottle]:
        return self.results

    def check_normalization(self, normalization: dict) -> None:
        return None

    def slugs(self) -> set[str]:
        return set()


class FilterResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        env = patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-only"})
        env.start()
        self.addCleanup(env.stop)

    def test_settings_are_loaded_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.yaml"
            path.write_text(
                DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
                    "model: deepseek/deepseek-v4.1-flash", "model: test/model"
                ),
                encoding="utf-8",
            )
            settings = load_ndr_config(path)
        self.assertEqual(settings.openrouter.model, "test/model")
        self.assertEqual(settings.openrouter.routing.only, ("together",))
        self.assertEqual(settings.generation.max_completion_tokens, 4096)
        self.assertEqual(FilterResolver(settings=settings).settings, settings)

    def test_local_catalog_supplies_full_card(self) -> None:
        bottle = crop()
        position = candidate(bottle, "first", "group", ["first", "second"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "catalog.csv"
            columns = [
                "Slug",
                *CSV_FIELDS.values(),
            ]
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                writer.writerow(
                    {
                        "Slug": "first",
                        "Выдержка или резерв": "резерв",
                    }
                )
            cards = load_cards(path)
        card = candidate_card(position, cards)
        self.assertEqual(card["aging_or_reserve"], "резерв")
        self.assertEqual(card["grapes"], "")
        self.assertTrue(card["reference_image_bytes"].startswith(b"\x89PNG"))

    def test_extra_body_reaches_request_and_overrides_generation(self) -> None:
        bottle = crop()
        candidates = [candidate(bottle, slug, "group", ["first", "second"])
                      for slug in ("first", "second")]
        with patch(PREDICT, return_value=model_response("first")) as model:
            FilterResolver()([BottleCandidates(bottle, candidates)])
        runtime = model.call_args.args[0]["runtime"]
        self.assertEqual(runtime["extra_body"], {"reasoning": {"enabled": False}})
        arguments = dict(system_prompt="test", api_content=[], output_format={"type": "json_schema"})
        payload = predictor.build_request_payload(runtime=runtime, **arguments)
        self.assertEqual(payload["reasoning"], {"enabled": False})
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["response_format"], arguments["output_format"])
        self.assertEqual(len(payload["messages"]), 2)
        self.assertNotIn("extra_body", payload)
        without_extra = {key: value for key, value in runtime.items() if key != "extra_body"}
        payload = predictor.build_request_payload(runtime=without_extra, **arguments)
        self.assertEqual(payload["reasoning"], runtime["generation"]["reasoning"])
        for key in ("model", "messages", "provider", "response_format"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                predictor.build_request_payload(runtime={**runtime, "extra_body": {key: None}}, **arguments)
        with self.assertRaises(TypeError):
            predictor.build_request_payload(runtime={**runtime, "extra_body": []}, **arguments)

    def test_payload_card_uses_box_crop_of_position(self) -> None:
        bottle = crop()
        position = candidate(bottle, "first", "first", ["first"])
        card = candidate_card(position)
        self.assertEqual(card["grapes"], "Пино нуар")
        self.assertEqual(card["aging_or_reserve"], "")
        self.assertTrue(card["reference_image_bytes"].startswith(b"\x89PNG"))

    def test_pipeline_runs_resolver_after_search(self) -> None:
        bottle = crop()
        top = candidate(bottle, "one", "", [])
        pipeline = Pipeline(
            normalizer=cast(Normalizer, NormalizerStub(bottle)),
            searcher=cast(Searcher, SearcherStub([BottleCandidates(bottle, [top])])),
            resolver=FilterResolver(),
            catalog=Catalog({}, "test"),
        )
        with patch(PREDICT, return_value=model_response("one")) as model:
            result = pipeline("image")
        answer = cast(MatchedBottle, result.resolution[0])
        self.assertEqual(answer.candidate.slug, "one")
        self.assertEqual(answer.source, "filter_v1")
        self.assertEqual(result.bottles[0].status, "matched")
        self.assertIn("resolve", result.timings)
        model.assert_called_once()

    def test_pipeline_keeps_search_rejections_and_order(self) -> None:
        bottle = crop()
        rejected = UnmatchedBottle(
            bottle, Rejection(SearchReason.NO_MATCH, "нет похожих")
        )
        pipeline = Pipeline(
            normalizer=cast(Normalizer, NormalizerStub(bottle)),
            searcher=cast(Searcher, SearcherStub([rejected])),
            resolver=FilterResolver(),
            catalog=Catalog({}, "тест"),
        )
        with patch(PREDICT) as model:
            result = pipeline("image")
        self.assertIs(result.resolution[0], rejected)
        self.assertEqual(result.matched, [])
        self.assertEqual(result.crops, [])
        outcome = result.bottles[0]
        self.assertEqual(
            (outcome.status, outcome.rejection), ("rejected", rejected.rejection)
        )
        self.assertEqual(rejected.rejection.stage, "search")
        model.assert_not_called()

    def test_pipeline_without_resolver_leaves_candidates(self) -> None:
        bottle = crop()
        top = candidate(bottle, "first", "group", ["first", "second"])
        second = candidate(bottle, "second", "group", ["first", "second"])
        pipeline = Pipeline(
            normalizer=cast(Normalizer, NormalizerStub(bottle)),
            searcher=cast(
                Searcher, SearcherStub([BottleCandidates(bottle, [top, second])])
            ),
            catalog=Catalog({}, "тест"),
            resolve=False,
        )
        with patch(PREDICT) as model:
            result = pipeline("image")
        found = cast(BottleCandidates, result.search[0])
        self.assertEqual([c.slug for c in found.candidates], ["first", "second"])
        self.assertEqual(result.resolution, [])
        self.assertEqual(result.bottles[0].status, "candidates")
        self.assertNotIn("resolve", result.timings)
        model.assert_not_called()

    def test_single_candidate_is_checked_by_model(self) -> None:
        bottle = crop()
        top = candidate(bottle, "one", "", [])
        for slug in ("one", "not_found"):
            with self.subTest(slug=slug), patch(PREDICT, return_value=model_response(slug)) as model:
                answer = FilterResolver()([BottleCandidates(bottle, [top])])[0]
            model.assert_called_once()
            self.assertEqual(len(model.call_args.args[0]["candidates"]), 1)
            if slug == "one":
                self.assertEqual(answer.candidate.slug, "one")
                self.assertEqual(answer.source, "filter_v1")
            else:
                self.assertIsInstance(answer, UnmatchedBottle)

    def test_all_five_candidates_are_sent_without_group_expansion(self) -> None:
        bottle = crop()
        bottle.box_crop[:] = 177
        candidates = [candidate(bottle, str(i), str(i), []) for i in range(5)]
        candidates[0].payload["group_slugs"] = ["0", "not_retrieved"]
        with patch(PREDICT, return_value=model_response("4")) as model:
            answer = FilterResolver()([BottleCandidates(bottle, candidates)])[0]
        self.assertIs(answer.candidate, candidates[4])
        self.assertEqual(answer.source, "filter_v1")
        request = model.call_args.args[0]
        self.assertEqual([c["slug"] for c in request["candidates"]], [str(i) for i in range(5)])
        self.assertNotIn("group", request)
        with Image.open(BytesIO(request["query"]["image_bytes"])) as query:
            self.assertEqual(query.getpixel((0, 0)), (177, 177, 177))

    def test_requests_run_concurrently_with_limit_and_keep_order(self) -> None:
        lock = threading.Lock()
        active = peak = 0

        def model(_request: dict) -> dict:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return model_response("first")

        found = []
        for index in range(NDR_CONCURRENCY + 8):
            bottle = crop(index)
            found.append(
                BottleCandidates(
                    bottle,
                    [
                        candidate(bottle, slug, "group", ["first", "second"])
                        for slug in ("first", "second")
                    ],
                )
            )
        with patch(PREDICT, side_effect=model):
            resolved = FilterResolver()(found)
        self.assertEqual(
            [answer.crop.index for answer in resolved], list(range(len(found)))
        )
        self.assertEqual(
            [cast(MatchedBottle, answer).candidate.slug for answer in resolved],
            ["first"] * len(found),
        )
        self.assertGreater(peak, 1)
        self.assertLessEqual(peak, NDR_CONCURRENCY)

    def test_semaphore_limits_separate_single_bottle_calls(self) -> None:
        lock = threading.Lock()
        active = peak = 0

        def model(_request: dict) -> dict:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.08)
            with lock:
                active -= 1
            return model_response("first")

        resolver = FilterResolver()
        bottle = crop()
        row = BottleCandidates(
            bottle,
            [
                candidate(bottle, slug, "group", ["first", "second"])
                for slug in ("first", "second")
            ],
        )
        with (
            patch(PREDICT, side_effect=model),
            ThreadPoolExecutor(max_workers=NDR_CONCURRENCY + 8) as executor,
        ):
            resolved = list(executor.map(resolver, [[row]] * (NDR_CONCURRENCY + 8)))
        self.assertEqual(len(resolved), NDR_CONCURRENCY + 8)
        self.assertGreater(peak, 1)
        self.assertLessEqual(peak, NDR_CONCURRENCY)

    def test_not_found_becomes_unmatched_bottle_with_observations(self) -> None:
        bottle = crop()
        candidates = [
            candidate(bottle, slug, "group", ["first", "second"])
            for slug in ("first", "second")
        ]
        with patch(PREDICT, return_value=model_response("not_found")):
            answer = cast(
                UnmatchedBottle,
                FilterResolver()([BottleCandidates(bottle, candidates)])[0],
            )
        self.assertIsInstance(answer, UnmatchedBottle)
        self.assertIs(answer.crop, bottle)
        self.assertEqual(answer.rejection.reason, "filter_not_found")
        self.assertEqual(answer.rejection.stage, "resolve")
        self.assertIn("year: 2023", answer.rejection.detail)

    def test_model_error_is_not_replaced_with_top_candidate(self) -> None:
        bottle = crop()
        candidates = [
            candidate(bottle, slug, "group", ["first", "second"])
            for slug in ("first", "second")
        ]
        failure = {
            "slug": None,
            "_status": "contract_error",
            "_error": "invalid JSON",
            "_trace": {"comparison_calls": []},
        }
        with (
            tempfile.TemporaryDirectory() as temp,
            patch(PREDICT, return_value=failure),
        ):
            trace_dir = Path(temp)
            with self.assertRaisesRegex(FilterError, "contract_error"):
                FilterResolver(trace_dir=trace_dir)(
                    [BottleCandidates(bottle, candidates)]
                )
            self.assertTrue((trace_dir / f"{bottle.uuid}.json").is_file())

    def test_duplicate_and_foreign_slugs_are_rejected(self) -> None:
        bottle = crop()
        top = candidate(bottle, "first", "", [])
        with patch(PREDICT) as model, self.assertRaises(FilterError):
            FilterResolver()([BottleCandidates(bottle, [top, top])])
        model.assert_not_called()
        with patch(PREDICT, return_value=model_response("foreign")), self.assertRaises(FilterError):
            FilterResolver()([BottleCandidates(bottle, [top])])

    def test_missing_openrouter_key_reports_env_file(self) -> None:
        bottle = crop()
        candidates = [
            candidate(bottle, slug, "group", ["first", "second"])
            for slug in ("first", "second")
        ]
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(predictor, "PROJECT_ROOT", Path(temp)),
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(FilterError, r"OPENROUTER_API_KEY.*\.env"),
        ):
            FilterResolver()([BottleCandidates(bottle, candidates)])

    def test_openrouter_key_is_loaded_from_project_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / ".env").write_text(
                "OPENROUTER_API_KEY=test-only\n", encoding="utf-8"
            )
            with (
                patch.object(predictor, "PROJECT_ROOT", Path(temp)),
                patch.dict(os.environ, {}, clear=True),
            ):
                predictor.load_openrouter_env()
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "test-only")

    def test_predictor_accepts_images_in_memory_with_strict_slug_enum(self) -> None:
        resolver = FilterResolver()
        request = {
            "query": {"image_bytes": b"\x89PNG\r\n\x1a\nquery"},
            "candidates": [
                    {
                        "slug": "first",
                        "vintage": "2023",
                        "reference_image_bytes": b"\x89PNG\r\n\x1a\nfirst",
                    },
                    {
                        "slug": "second",
                        "vintage": "",
                        "reference_image_bytes": b"\x89PNG\r\n\x1a\nsecond",
                    },
                ],
            "prompts": resolver.prompts,
            "runtime": {
                "model": "test/model",
                "api_key_env": "NDR_OFFLINE_TEST_KEY",
                "image_detail": "original",
            },
        }
        selected = model_response("second")["_trace"]["comparison_calls"][0]["selected"]
        with (
            patch.dict(os.environ, {"NDR_OFFLINE_TEST_KEY": "test-only"}),
            patch.object(
                predictor,
                "call_model",
                return_value={
                    "status": "ok",
                    "error": None,
                    "selected": selected,
                    "trace": {},
                },
            ) as call,
        ):
            response = predictor.predict(request)
        self.assertEqual(response["slug"], "second")
        self.assertEqual(call.call_count, 1)
        schema = call.call_args.kwargs["output_format"]["json_schema"]["schema"]
        self.assertEqual(
            schema["properties"]["slug"]["enum"], ["first", "second", "not_found"]
        )
        self.assertEqual(
            len(
                [
                    part
                    for part in call.call_args.kwargs["api_content"]
                    if part["type"] == "image_url"
                ]
            ),
            3,
        )
        self.assertTrue(
            all(
                part["image"]["path"] is None
                for part in call.call_args.kwargs["trace_content"]
                if part["type"] == "image_url"
            )
        )


    def test_pipeline_selects_second_level_from_search_mode(self) -> None:
        for grouped in (False, True):
            with self.subTest(grouped=grouped):
                searcher = Mock(spec=VisSearcher)
                searcher.cfg = load_search_config()
                if grouped:
                    searcher.cfg.search = GroupSearch(mode="groups", top_groups=5, group_threshold=0.7)
                searcher.slugs.return_value = set()
                pipeline = Pipeline(
                    normalizer=cast(Normalizer, NormalizerStub(crop())),
                    searcher=searcher,
                    catalog=Catalog({}, "test"),
                )
                if grouped:
                    from vinishko.pipeline.steps.near_duplicates import NearDuplicateResolver
                    self.assertIsInstance(pipeline.resolver, NearDuplicateResolver)
                else:
                    self.assertIsInstance(pipeline.resolver, FilterResolver)
                disabled = Pipeline(
                    normalizer=cast(Normalizer, NormalizerStub(crop())),
                    searcher=searcher,
                    catalog=Catalog({}, "test"),
                    resolve=False,
                )
                self.assertIsNone(disabled.resolver)

    def test_search_top_n_passes_five_candidates_and_checks_best_score(self) -> None:
        searcher = object.__new__(VisSearcher)
        searcher.cfg = load_search_config()
        self.assertEqual(searcher.cfg.top_k, 5)
        self.assertEqual(searcher.cfg.search.mode, "top_n")
        searcher.measure = "cosine"
        bottle = crop()
        candidates = [candidate(bottle, str(i), str(i), []) for i in range(5)]
        hits = [(0.8 - i * 0.1, {"slug": str(i)}, {}) for i in range(5)]
        with patch.object(searcher, "_hits", return_value=hits), patch.object(searcher, "_candidate", side_effect=candidates):
            result = searcher._search(bottle, {})
        self.assertEqual(result.candidates, candidates)
        self.assertIsInstance(searcher._top_n(bottle, [(0.69, {}, {})], searcher.cfg.search), UnmatchedBottle)

    def test_final_metrics_use_exact_slug_and_count_filter_calls(self) -> None:
        bottle = crop()
        chosen = candidate(bottle, "chosen", "shared", ["chosen", "expected"])
        chosen.payload["group_slugs"] = ["chosen", "expected"]
        found = BottleCandidates(bottle, [chosen])
        verdict = MatchedBottle(bottle, chosen, "filter_v1", {})
        result = PipelineResult([bottle], search=[found], resolution=[verdict])
        rows = []
        for expected in ("chosen", "expected", ""):
            search_row = SearchRow("test.jpg", expected, bool(expected), "candidates", 1, "", "chosen", 0.8, 1, 1, True, 0.1)
            row = e2e_row(search_row, result)
            self.assertEqual(row.correct, expected == "chosen")
            if expected:
                self.assertTrue(row.group_correct)
            rows.append(row)
        rejected = UnmatchedBottle(bottle, Rejection(SearchReason.NO_MATCH, "test"))
        result.resolution = [rejected]
        empty = SearchRow("empty.jpg", "", False, "no_match", 1, "", "", None, None, None, False, 0.1)
        self.assertTrue(e2e_row(empty, result).correct)
        self.assertEqual(e2e_metrics(rows, [])["resolved_by_model"], 3)

    def test_evaluator_keeps_filter_trace(self) -> None:
        bottle = crop()
        search_row = SearchRow("test.jpg", "one", True, "candidates", 1, "", "one", 0.8, 1, 1, False, 0.1)
        result = PipelineResult([bottle], search=[BottleCandidates(bottle, [candidate(bottle, "one", "", [])])])
        pipeline = Mock()
        pipeline.searcher = None
        pipeline.resolver = FilterResolver()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("vinishko.pipeline.evaluate_pipeline.evaluate_image", return_value=(search_row, result)),
            patch("vinishko.pipeline.evaluate_pipeline.dump_image"),
        ):
            run(pipeline, [("test.jpg", "one")], Path(directory), Path(directory), {"one"}, {}, "all", False)
            self.assertEqual(pipeline.resolver.trace_dir, Path(directory) / "dumps/test/resolve")

    def test_api_health_and_metrics_recognize_filter(self) -> None:
        from fastapi.testclient import TestClient
        from vinishko.app.main import create_app
        from vinishko.e2e import row_of, metrics

        pipeline = Mock()
        pipeline.searcher = None
        pipeline.resolver = FilterResolver()
        pipeline.catalog = Catalog({}, "test")
        with TestClient(create_app(pipeline)) as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["resolver"], pipeline.resolver.settings.openrouter.model)
        body = {"bottles": [{"status": "matched", "match": {"slug": "chosen", "source": "filter_v1"}}]}
        row = row_of("test.jpg", "chosen", 200, body, 0.1)
        self.assertTrue(row.correct)
        self.assertEqual(metrics([row])["resolved_by_model"], 1)


class OpenRouterProxyTests(unittest.TestCase):
    def test_custom_proxy_and_default_transport(self) -> None:
        for proxy in ("", "   ", " http://127.0.0.1:8080 "):
            with (
                self.subTest(proxy=proxy),
                patch.dict(os.environ, {"openrouter_http_proxy": proxy}),
                patch.object(predictor.urllib.request, "build_opener") as build_opener,
                patch.object(predictor.urllib.request, "urlopen") as urlopen,
            ):
                offline = predictor.urllib.error.URLError("offline")
                build_opener.return_value.open.side_effect = offline
                urlopen.side_effect = offline
                result = predictor.call_model(
                    runtime={"model": "test/model", "api_base": "https://openrouter.ai/api/v1"},
                    api_key="test-only",
                    system_prompt="test",
                    api_content=[],
                    trace_content=[],
                    output_format={"json_schema": {"schema": {}}},
                    validator=lambda value: value,
                )
                self.assertEqual(result["status"], "predictor_error")
                if proxy.strip():
                    handler = build_opener.call_args.args[0]
                    self.assertEqual(handler.proxies, {"http": proxy.strip(), "https": proxy.strip()})
                    build_opener.return_value.open.assert_called_once()
                    urlopen.assert_not_called()
                else:
                    build_opener.assert_not_called()
                    urlopen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
