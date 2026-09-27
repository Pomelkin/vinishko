"""Офлайн-проверки второго уровня: без qdrant, S3 и модели. Запуск из корня: python -m unittest vinishko.pipeline.steps.near_duplicates.test_resolve -v"""

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
from unittest.mock import patch

import numpy as np
from PIL import Image

from vinishko.pipeline.catalog import Catalog
from vinishko.pipeline.pipeline import Pipeline
from vinishko.pipeline.pipeline import Searcher
from vinishko.pipeline.steps.near_duplicates import predictor
from vinishko.pipeline.steps.near_duplicates.configs import DEFAULT_CONFIG
from vinishko.pipeline.steps.near_duplicates.configs import (
    load_config as load_ndr_config,
)
from vinishko.pipeline.steps.near_duplicates.resolve import CSV_FIELDS
from vinishko.pipeline.steps.near_duplicates.resolve import NDR_CONCURRENCY
from vinishko.pipeline.steps.near_duplicates.resolve import NearDuplicateError
from vinishko.pipeline.steps.near_duplicates.resolve import NearDuplicateResolver
from vinishko.pipeline.steps.near_duplicates.resolve import candidate_card
from vinishko.pipeline.steps.near_duplicates.resolve import load_cards
from vinishko.pipeline.steps.normalization.normalize import Normalizer
from vinishko.pipeline.steps.vis_searcher.search import SearchReason
from vinishko.pipeline.structs import BottleCandidates
from vinishko.pipeline.structs import BottleCrop
from vinishko.pipeline.structs import Candidate
from vinishko.pipeline.structs import MatchedBottle
from vinishko.pipeline.structs import Rejection
from vinishko.pipeline.structs import UnmatchedBottle


PREDICT = "vinishko.pipeline.steps.near_duplicates.resolve.predict"


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
            "group": group,
            "group_slugs": members,
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


class NearDuplicateResolverTests(unittest.TestCase):
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
        self.assertEqual(NearDuplicateResolver(settings=settings).settings, settings)

    def test_local_catalog_supplies_full_card(self) -> None:
        bottle = crop()
        position = candidate(bottle, "first", "group", ["first", "second"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "catalog.csv"
            columns = [
                "Slug",
                "image_filename",
                "near_duplicate_group_slug",
                *CSV_FIELDS.values(),
            ]
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                writer.writerow(
                    {
                        "Slug": "first",
                        "image_filename": "original.webp",
                        "near_duplicate_group_slug": "group",
                        "Выдержка или резерв": "резерв",
                    }
                )
            cards = load_cards(path)
        card = candidate_card(position, cards)
        self.assertEqual(card["aging_or_reserve"], "резерв")
        self.assertEqual(card["grapes"], "")
        self.assertTrue(card["reference_image_bytes"].startswith(b"\x89PNG"))

    def test_payload_card_uses_box_crop_of_position(self) -> None:
        bottle = crop()
        position = candidate(bottle, "first", "first", ["first"])
        card = candidate_card(position)
        self.assertEqual(card["grapes"], "Пино нуар")
        self.assertEqual(card["aging_or_reserve"], "")
        self.assertTrue(card["reference_image_bytes"].startswith(b"\x89PNG"))

    def test_pipeline_runs_resolver_after_search(self) -> None:
        bottle = crop()
        top = candidate(bottle, "one", "one", ["one"])
        pipeline = Pipeline(
            normalizer=cast(Normalizer, NormalizerStub(bottle)),
            searcher=cast(Searcher, SearcherStub([BottleCandidates(bottle, [top])])),
            resolver=NearDuplicateResolver(),
            catalog=Catalog({}, "тест"),
        )
        with patch(PREDICT) as model:
            result = pipeline("image")
        answer = cast(MatchedBottle, result.resolution[0])
        self.assertIsInstance(answer, MatchedBottle)
        self.assertEqual(answer.candidate.slug, "one")
        self.assertEqual(answer.source, "vector")
        self.assertEqual([m.candidate.slug for m in result.matched], ["one"])
        self.assertEqual(result.crops, [bottle])
        outcome = result.bottles[0]
        self.assertEqual(
            (outcome.status, outcome.uuid, outcome.crop),
            ("matched", bottle.uuid, bottle),
        )
        self.assertIs(outcome.match, answer)
        self.assertIn("resolve", result.timings)
        model.assert_not_called()

    def test_pipeline_keeps_search_rejections_and_order(self) -> None:
        bottle = crop()
        rejected = UnmatchedBottle(
            bottle, Rejection(SearchReason.NO_MATCH, "нет похожих")
        )
        pipeline = Pipeline(
            normalizer=cast(Normalizer, NormalizerStub(bottle)),
            searcher=cast(Searcher, SearcherStub([rejected])),
            resolver=NearDuplicateResolver(),
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

    def test_singleton_group_returns_top_without_model(self) -> None:
        bottle = crop()
        top = candidate(bottle, "one", "one", ["one"])
        with patch(PREDICT) as model:
            answer = cast(
                MatchedBottle,
                NearDuplicateResolver()([BottleCandidates(bottle, [top])])[0],
            )
        self.assertIsInstance(answer, MatchedBottle)
        self.assertEqual(answer.candidate.slug, "one")
        self.assertEqual(answer.source, "vector")
        self.assertEqual(answer.checklist, {})
        model.assert_not_called()

    def test_group_sends_box_crop_and_returns_selected_slug(self) -> None:
        bottle = crop()
        bottle.box_crop[:] = 177
        top = candidate(bottle, "first", "group", ["first", "second"])
        second = candidate(bottle, "second", "group", ["first", "second"])
        outsider = candidate(bottle, "outsider", "other", ["outsider"])
        with patch(PREDICT, return_value=model_response("second")) as model:
            answer = cast(
                MatchedBottle,
                NearDuplicateResolver()(
                    [BottleCandidates(bottle, [top, second, outsider])]
                )[0],
            )
        self.assertIsInstance(answer, MatchedBottle)
        self.assertEqual(answer.candidate.slug, "second")
        self.assertEqual(answer.source, "ndr_v5")
        self.assertEqual(answer.checklist["year"]["observation"], "2023")
        request = model.call_args.args[0]
        self.assertEqual(
            [c["slug"] for c in request["group"]["candidates"]], ["first", "second"]
        )
        self.assertEqual(request["group"]["candidates"][0]["grapes"], "Пино нуар")
        self.assertTrue(request["query"]["image_bytes"].startswith(b"\x89PNG"))
        with Image.open(BytesIO(request["query"]["image_bytes"])) as query:
            self.assertEqual(query.size, (4, 4))
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
            resolved = NearDuplicateResolver()(found)
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

        resolver = NearDuplicateResolver()
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
                NearDuplicateResolver()([BottleCandidates(bottle, candidates)])[0],
            )
        self.assertIsInstance(answer, UnmatchedBottle)
        self.assertIs(answer.crop, bottle)
        self.assertEqual(answer.rejection.reason, "near_duplicate_not_found")
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
            with self.assertRaisesRegex(NearDuplicateError, "contract_error"):
                NearDuplicateResolver(trace_dir=trace_dir)(
                    [BottleCandidates(bottle, candidates)]
                )
            self.assertTrue((trace_dir / f"{bottle.uuid}.json").is_file())

    def test_incomplete_group_fails_before_model_call(self) -> None:
        bottle = crop()
        top = candidate(bottle, "first", "group", ["first", "second"])
        with (
            patch(PREDICT) as model,
            self.assertRaisesRegex(NearDuplicateError, "groups"),
        ):
            NearDuplicateResolver()([BottleCandidates(bottle, [top])])
        model.assert_not_called()

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
            self.assertRaisesRegex(NearDuplicateError, r"OPENROUTER_API_KEY.*\.env"),
        ):
            NearDuplicateResolver()([BottleCandidates(bottle, candidates)])

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
        resolver = NearDuplicateResolver()
        request = {
            "query": {"image_bytes": b"\x89PNG\r\n\x1a\nquery"},
            "group": {
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
                ]
            },
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


if __name__ == "__main__":
    unittest.main()
