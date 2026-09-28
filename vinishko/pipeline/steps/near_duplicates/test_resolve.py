"""Офлайн-проверки второго уровня: без qdrant, S3 и модели. Запуск из корня: python -m unittest vinishko.pipeline.steps.near_duplicates.test_resolve -v"""

import csv
import email.message
import io
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.response
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import ClassVar
from typing import cast
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
from PIL import Image

from vinishko.pipeline.catalog import Catalog
from vinishko.pipeline.pipeline import Pipeline
from vinishko.pipeline.pipeline import Searcher
from vinishko.pipeline.steps.near_duplicates import predictor
from vinishko.pipeline.steps.near_duplicates.configs import DEFAULT_CONFIG
from vinishko.pipeline.steps.near_duplicates.models import nearest_output_model
from vinishko.pipeline.steps.near_duplicates.models import response_format
from vinishko.pipeline.steps.near_duplicates.models import validate_output
from vinishko.pipeline.steps.near_duplicates.configs import (
    load_config as load_ndr_config,
)
from vinishko.pipeline.steps.near_duplicates.resolve import CSV_FIELDS
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
LIMIT = load_ndr_config().execution.concurrency


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
            "differences": {"observation": "none"},
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


def model_by_task(choose: Callable[[str, list[str]], str]) -> Callable[[dict], dict]:
    """Подменённый predict: ответ по задаче вызова и slug его позиций."""

    def model(request: dict) -> dict:
        slugs = [card["slug"] for card in request["candidates"]]
        return model_response(choose(request["task"], slugs))

    return model


def group(bottle: BottleCrop, name: str, slugs: list[str]) -> list[Candidate]:
    """Все позиции одной группы кандидатов."""
    return [candidate(bottle, slug, name, slugs) for slug in slugs]


def calls(model: MagicMock) -> list[tuple[str, list[str]]]:
    """Вызовы подменённого predict: задача и slug позиций, по порядку."""
    return [
        (call.args[0]["task"], [card["slug"] for card in call.args[0]["candidates"]])
        for call in model.call_args_list
    ]


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
        self.assertEqual(
            (settings.generation.reasoning_effort.group, settings.generation.reasoning_effort.final),
            ("none", "low"),
        )

    def test_reasoning_effort_is_set_per_round(self) -> None:
        settings = load_ndr_config()
        generation = settings.generation.model_copy(
            update={
                "reasoning_effort": settings.generation.reasoning_effort.model_copy(
                    update={"group": None, "final": "low"}
                )
            }
        )
        self.assertNotIn("reasoning", generation.request_payload("group"))
        self.assertEqual(
            generation.request_payload("final")["reasoning"], {"effort": "low", "exclude": False}
        )
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
        self.assertTrue(card["reference_image_bytes"].startswith(b"\xff\xd8\xff"))

    def test_payload_card_uses_box_crop_of_position(self) -> None:
        bottle = crop()
        position = candidate(bottle, "first", "first", ["first"])
        card = candidate_card(position)
        self.assertEqual(card["grapes"], "Пино нуар")
        self.assertEqual(card["aging_or_reserve"], "")
        self.assertTrue(card["reference_image_bytes"].startswith(b"\xff\xd8\xff"))

    def test_pipeline_runs_resolver_after_search(self) -> None:
        bottle = crop()
        top = candidate(bottle, "one", "one", ["one"])
        pipeline = Pipeline(
            normalizer=cast(Normalizer, NormalizerStub(bottle)),
            searcher=cast(Searcher, SearcherStub([BottleCandidates(bottle, [top])])),
            resolver=NearDuplicateResolver(),
            catalog=Catalog({}, "тест"),
        )
        with patch(PREDICT, return_value=model_response("one")) as model:
            result = pipeline("image")
        answer = cast(MatchedBottle, result.resolution[0])
        self.assertIsInstance(answer, MatchedBottle)
        self.assertEqual(answer.candidate.slug, "one")
        self.assertEqual(answer.source, "final")
        self.assertEqual([m.candidate.slug for m in result.matched], ["one"])
        self.assertEqual(result.crops, [bottle])
        outcome = result.bottles[0]
        self.assertEqual(
            (outcome.status, outcome.uuid, outcome.crop),
            ("matched", bottle.uuid, bottle),
        )
        self.assertIs(outcome.match, answer)
        self.assertIn("resolve", result.timings)
        self.assertEqual(calls(model), [("final", ["one"])])

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

    def test_lone_singleton_is_verified_by_final_call(self) -> None:
        bottle = crop()
        found = [BottleCandidates(bottle, [candidate(bottle, "one", "one", ["one"])])]
        with patch(PREDICT, return_value=model_response("one")) as model:
            answer = cast(MatchedBottle, NearDuplicateResolver()(found)[0])
        self.assertIsInstance(answer, MatchedBottle)
        self.assertEqual(answer.candidate.slug, "one")
        self.assertEqual(answer.source, "final")
        self.assertEqual(answer.checklist["year"]["observation"], "2023")
        self.assertEqual(calls(model), [("final", ["one"])])
        with patch(PREDICT, return_value=model_response("not_found")):
            rejected = cast(UnmatchedBottle, NearDuplicateResolver()(found)[0])
        self.assertIsInstance(rejected, UnmatchedBottle)
        self.assertEqual(rejected.rejection.stage, "resolve")
        self.assertIn("финалистов one", rejected.rejection.detail)

    def test_group_winner_meets_singleton_in_final(self) -> None:
        bottle = crop()
        bottle.box_crop[:] = 177
        members = group(bottle, "group", ["first", "second"])
        outsider = candidate(bottle, "outsider", "other", ["outsider"])
        with patch(
            PREDICT, side_effect=model_by_task(lambda task, slugs: "second")
        ) as model:
            answer = cast(
                MatchedBottle,
                NearDuplicateResolver()([BottleCandidates(bottle, [*members, outsider])])[
                    0
                ],
            )
        self.assertIsInstance(answer, MatchedBottle)
        self.assertEqual(answer.candidate.slug, "second")
        self.assertIs(answer.candidate, members[1])
        self.assertEqual(answer.source, "final")
        self.assertEqual(
            calls(model),
            [("group", ["first", "second"]), ("final", ["second", "outsider"])],
        )
        requests = [call.args[0] for call in model.call_args_list]
        self.assertEqual(requests[0]["candidates"][0]["grapes"], "Пино нуар")
        self.assertIs(requests[0]["query"]["image_bytes"], requests[1]["query"]["image_bytes"])
        self.assertTrue(requests[0]["query"]["image_bytes"].startswith(b"\xff\xd8\xff"))
        with Image.open(BytesIO(requests[0]["query"]["image_bytes"])) as query:
            self.assertEqual((query.format, query.size), ("JPEG", (4, 4)))
            pixel = cast(tuple[int, int, int], query.getpixel((0, 0)))
            self.assertTrue(all(abs(value - 177) <= 2 for value in pixel))

    def test_single_group_winner_skips_final(self) -> None:
        bottle = crop()
        members = group(bottle, "group", ["first", "second"])
        with patch(PREDICT, return_value=model_response("second")) as model:
            answer = cast(
                MatchedBottle,
                NearDuplicateResolver()([BottleCandidates(bottle, members)])[0],
            )
        self.assertEqual(answer.candidate.slug, "second")
        self.assertEqual(answer.source, "group")
        self.assertEqual(answer.checklist["year"]["observation"], "2023")
        self.assertEqual(calls(model), [("group", ["first", "second"])])

    def test_final_chooses_among_winners_of_every_group(self) -> None:
        bottle = crop()
        first = group(bottle, "a", ["a1", "a2"])
        second = group(bottle, "b", ["b1", "b2"])
        rejected = group(bottle, "c", ["c1", "c2"])
        picks = {"a1": "a1", "b1": "b2", "c1": "not_found", "a1+b2": "b2"}
        with (
            tempfile.TemporaryDirectory() as temp,
            patch(
                PREDICT,
                side_effect=model_by_task(
                    lambda task, slugs: picks["+".join(slugs) if task == "final" else slugs[0]]
                ),
            ) as model,
        ):
            answer = cast(
                MatchedBottle,
                NearDuplicateResolver(trace_dir=Path(temp))(
                    [BottleCandidates(bottle, [*first, *second, *rejected])]
                )[0],
            )
            traces = sorted(p.name for p in (Path(temp) / bottle.uuid).iterdir())
        self.assertEqual(answer.candidate.slug, "b2")
        self.assertEqual(answer.source, "final")
        self.assertEqual(
            sorted(calls(model)),
            [
                ("final", ["a1", "b2"]),
                ("group", ["a1", "a2"]),
                ("group", ["b1", "b2"]),
                ("group", ["c1", "c2"]),
            ],
        )
        self.assertEqual(calls(model)[-1], ("final", ["a1", "b2"]))
        self.assertEqual(traces, ["final.json", "group1.json", "group2.json", "group3.json"])

    def test_groups_rejected_by_model_leave_no_final(self) -> None:
        bottle = crop()
        first = group(bottle, "a", ["a1", "a2"])
        second = group(bottle, "b", ["b1", "b2"])
        with patch(
            PREDICT,
            side_effect=model_by_task(
                lambda task, slugs: "b1" if slugs[0] == "b1" else "not_found"
            ),
        ) as model:
            answer = cast(
                MatchedBottle,
                NearDuplicateResolver()([BottleCandidates(bottle, [*first, *second])])[0],
            )
        self.assertEqual(answer.candidate.slug, "b1")
        self.assertEqual(answer.source, "group")
        self.assertEqual(len(calls(model)), 2)
        with patch(PREDICT, return_value=model_response("not_found")) as model:
            rejected = cast(
                UnmatchedBottle,
                NearDuplicateResolver()([BottleCandidates(bottle, [*first, *second])])[0],
            )
        self.assertIsInstance(rejected, UnmatchedBottle)
        self.assertIn("каждой из 2 групп", rejected.rejection.detail)
        self.assertEqual(sorted(task for task, _ in calls(model)), ["group", "group"])

    def test_final_not_found_rejects_bottle(self) -> None:
        bottle = crop()
        members = group(bottle, "group", ["first", "second"])
        outsider = candidate(bottle, "outsider", "other", ["outsider"])
        with patch(
            PREDICT,
            side_effect=model_by_task(
                lambda task, slugs: "not_found" if task == "final" else "first"
            ),
        ):
            answer = cast(
                UnmatchedBottle,
                NearDuplicateResolver()([BottleCandidates(bottle, [*members, outsider])])[
                    0
                ],
            )
        self.assertIsInstance(answer, UnmatchedBottle)
        self.assertEqual(answer.rejection.reason, "near_duplicate_not_found")
        self.assertIn("финалистов first, outsider", answer.rejection.detail)

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
        for index in range(LIMIT + 8):
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
        self.assertLessEqual(peak, LIMIT)

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
            ThreadPoolExecutor(max_workers=LIMIT + 8) as executor,
        ):
            resolved = list(executor.map(resolver, [[row]] * (LIMIT + 8)))
        self.assertEqual(len(resolved), LIMIT + 8)
        self.assertGreater(peak, 1)
        self.assertLessEqual(peak, LIMIT)

    def test_concurrency_comes_from_settings(self) -> None:
        lock = threading.Lock()
        active = peak = 0

        def model(request: dict) -> dict:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return model_response(request["candidates"][0]["slug"])

        settings = load_ndr_config()
        settings = settings.model_copy(
            update={"execution": settings.execution.model_copy(update={"concurrency": 2})}
        )
        bottle = crop()
        found = [
            BottleCandidates(
                bottle,
                [
                    *group(bottle, "a", ["a1", "a2"]),
                    *group(bottle, "b", ["b1", "b2"]),
                    *group(bottle, "c", ["c1", "c2"]),
                    *group(bottle, "d", ["d1", "d2"]),
                ],
            )
        ]
        with patch(PREDICT, side_effect=model):
            NearDuplicateResolver(settings=settings)(found)
        self.assertEqual(peak, 2)

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
            self.assertTrue((trace_dir / bottle.uuid / "group1.json").is_file())

    def test_incomplete_group_fails_before_model_call(self) -> None:
        bottle = crop()
        top = candidate(bottle, "first", "group", ["first", "second"])
        with (
            patch(PREDICT) as model,
            self.assertRaisesRegex(NearDuplicateError, "groups"),
        ):
            NearDuplicateResolver()([BottleCandidates(bottle, [top])])
        model.assert_not_called()
        lower = candidate(bottle, "b1", "b", ["b1", "b2"])
        complete = group(bottle, "a", ["a1", "a2"])
        with (
            patch(PREDICT) as model,
            self.assertRaisesRegex(NearDuplicateError, "группа b"),
        ):
            NearDuplicateResolver()([BottleCandidates(bottle, [*complete, lower])])
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
            "task": "group",
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
        self.assertEqual(list(schema["properties"]), ["checklist", "slug"])
        self.assertEqual(
            list(schema["$defs"]["SelectionChecklist"]["properties"]),
            ["differences", "maker", "profile", "year"],
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
        group_prompt = call.call_args.kwargs["system_prompt"]
        self.assertIn("complete supplied group", group_prompt)
        self.assertIn("only then treat it as a redesign of the same product", group_prompt)
        self.assertIn("a candidate slug requires `none`", group_prompt)
        self.assertIn("ELEMENT 1 (first): an exact match with 2023", group_prompt)
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
            response = predictor.predict(request | {"task": "final"})
        self.assertEqual(response["slug"], "second")
        final_prompt = call.call_args.kwargs["system_prompt"]
        self.assertIn("different design family", final_prompt)
        self.assertNotIn("may refresh the label", final_prompt)
        self.assertIn("ULTRA CUVEE equals УЛЬТРА КЮВЕ", final_prompt)
        self.assertNotIn("complete supplied group", final_prompt)
        self.assertIn("ELEMENT 2 (second): an exact vintage match is not required", final_prompt)
        self.assertEqual(
            call.call_args.kwargs["output_format"]["json_schema"]["name"],
            "ndr_select_final",
        )
        self.assertIn(
            "same product as QUERY", call.call_args.kwargs["api_content"][-1]["text"]
        )
        unknown = predictor.predict(request | {"task": "rerank"})
        self.assertEqual(
            (unknown["_status"], unknown["_error"]),
            ("predictor_error", "unknown task 'rerank'"),
        )


def http_error(code: int, reason: str, retry_after: str | None = None) -> urllib.error.HTTPError:
    """Ответ OpenRouter с ошибкой, как его отдаёт urllib."""
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    body = json.dumps({"error": {"code": code, "message": reason}}).encode()
    return urllib.error.HTTPError(
        "https://openrouter.ai/api/v1/chat/completions", code, reason, headers, io.BytesIO(body)
    )


def http_ok(payload: dict) -> urllib.response.addinfourl:
    """Успешный ответ OpenRouter с телом payload."""
    return urllib.response.addinfourl(
        io.BytesIO(json.dumps(payload).encode()),
        email.message.Message(),
        "https://openrouter.ai/api/v1/chat/completions",
        200,
    )


def completion(slug: str) -> dict:
    """Тело chat completion с валидным ответом модели."""
    content = {
        "checklist": {
            key: {"observation": "none" if key == "differences" else "ok"}
            for key in ("differences", "maker", "profile", "year")
        },
        "slug": slug,
    }
    return {"choices": [{"index": 0, "finish_reason": "stop", "message": {"content": json.dumps(content)}}]}


class RetryTests(unittest.TestCase):
    """Повторы запроса к модели через tenacity: сеть подменена, паузы не ждём."""

    def call(self, retries: int = 3) -> dict:
        output = nearest_output_model(["first", "second"])
        return predictor.call_model(
            runtime={
                "api_base": "https://openrouter.ai/api/v1",
                "model": "test/model",
                "timeout": 60,
                "retries": retries,
                "generation": {},
                "provider": {},
            },
            api_key="test-only",
            system_prompt="prompt",
            api_content=[{"type": "text", "text": "query"}],
            trace_content=[{"type": "text", "text": "query"}],
            output_format=response_format("ndr_select_group", output),
            validator=lambda payload: validate_output(payload, output),
        )

    def test_throttled_and_transient_answers_are_retried(self) -> None:
        answers = [
            http_error(429, "Too Many Requests", retry_after="2"),
            http_error(503, "Service Unavailable"),
            http_ok(completion("second")),
        ]
        with (
            patch("urllib.request.urlopen", side_effect=answers) as urlopen,
            patch("time.sleep") as sleep,
        ):
            result = self.call()
        self.assertEqual((result["status"], result["selected"]["slug"]), ("ok", "second"))
        self.assertEqual(urlopen.call_count, 3)
        attempts = result["trace"]["retry_attempts"]
        self.assertEqual([a["response_status"] for a in attempts], [429, 503])
        self.assertGreaterEqual(sleep.call_args_list[0].args[0], 2.0)
        self.assertLessEqual(sleep.call_args_list[1].args[0], 2.0)

    def test_retries_stop_after_limit(self) -> None:
        with (
            patch(
                "urllib.request.urlopen",
                side_effect=lambda *_a, **_k: (_ for _ in ()).throw(http_error(429, "Too Many Requests")),
            ) as urlopen,
            patch("time.sleep"),
        ):
            result = self.call(retries=2)
        self.assertEqual(result["status"], "predictor_error")
        self.assertEqual(result["error"], "HTTPError 429: Too Many Requests; попыток 3")
        self.assertEqual(urlopen.call_count, 3)

    def test_client_error_is_not_retried(self) -> None:
        with (
            patch("urllib.request.urlopen", side_effect=[http_error(400, "Bad Request")]) as urlopen,
            patch("time.sleep") as sleep,
        ):
            result = self.call()
        self.assertEqual(result["error"], "HTTPError 400: Bad Request")
        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()

    def test_rate_limit_inside_ok_body_is_retried(self) -> None:
        answers = [
            http_ok({"error": {"code": 429, "message": "provider rate limited"}}),
            http_ok(completion("first")),
        ]
        with patch("urllib.request.urlopen", side_effect=answers), patch("time.sleep"):
            result = self.call()
        self.assertEqual((result["status"], result["selected"]["slug"]), ("ok", "first"))
        self.assertEqual(len(result["trace"]["retry_attempts"]), 1)


if __name__ == "__main__":
    unittest.main()
