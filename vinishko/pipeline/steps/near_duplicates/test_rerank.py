"""Офлайн-проверки перехода от группы Qdrant к NDR v5."""

import csv
import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from vinishko.pipeline.steps.near_duplicates import predictor
from vinishko.pipeline.steps.near_duplicates.rerank import CSV_FIELDS, NearDuplicateError, NearDuplicateReranker, candidate_card, load_cards
from vinishko.pipeline.steps.vis_searcher.storage import S3Store
from vinishko.pipeline.steps.vis_searcher.catalog import CollectionInfo
from vinishko.pipeline.steps.vis_searcher.configs import load_config
from vinishko.pipeline.steps.vis_searcher.search import VisSearcher
from vinishko.pipeline.pipeline import Pipeline
from vinishko.pipeline.structs import BottleCandidates, BottleCrop, Candidate, UnmatchedBottle


def crop() -> BottleCrop:
    """Минимальная бутылка с валидным RGB-кропом."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return BottleCrop(1, 0.9, [], [], 0.0, image, {}, image, {})


def candidate(bottle: BottleCrop, slug: str, group: str, members: list[str]) -> Candidate:
    """Кандидат с метаданными Qdrant, которые использует адаптер."""
    return Candidate(slug, 0.8, np.zeros((4, 4, 3), dtype=np.uint8), bottle, group, True, {
        "slug": slug,
        "group": group,
        "group_slugs": members,
        "source_image": f"{slug}.webp",
        "grape": "Пино нуар",
        "vintage": "2023",
    })


def model_response(slug: str) -> dict:
    """Валидный ответ от уже мокнутого predictor.predict."""
    selected = {"slug": slug, "checklist": {
        "maker": {"observation": "совпадает"},
        "profile": {"observation": "совпадает"},
        "year": {"observation": "2023"},
    }}
    return {"slug": slug, "_status": "ok", "_error": None, "_trace": {"comparison_calls": [{"selected": selected}]}}


class FakeStore:
    """Эталоны в памяти без сетевых вызовов."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def get_bytes(self, name: str) -> bytes:
        """Запоминает запрошенный source_image."""
        self.names.append(name)
        return b"RIFF\x00\x00\x00\x00WEBPtest"


class NearDuplicateRerankerTests(unittest.TestCase):
    def test_local_catalog_supplies_full_v5_card(self) -> None:
        bottle = crop()
        position = candidate(bottle, "first", "group", ["first", "second"])
        store = FakeStore()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "catalog.csv"
            columns = ["Slug", "image_filename", "near_duplicate_group_slug", *CSV_FIELDS.values()]
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                writer.writerow({"Slug": "first", "image_filename": "original.webp", "near_duplicate_group_slug": "group", "Выдержка или резерв": "резерв"})
            cards = load_cards(path)
        card = candidate_card(position, store, cards)
        self.assertEqual(card["aging_or_reserve"], "резерв")
        self.assertEqual(card["grapes"], "")
        self.assertEqual(store.names, ["original.webp"])

    def test_query_only_search_uses_local_source_image_without_s3(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Image.new("RGB", (4, 4), "red").save(root / "source.png")
            cfg = load_config()
            cfg.images = None
            cfg.reference_images.dir = root
            files = SimpleNamespace(repo=cfg.model, revision="revision", embed_dim=4, input_size=(512, 512))
            info = CollectionInfo(cfg.qdrant.collection, 4, 1, cfg.model, "revision", (512, 512))
            payload = {"slug": "first", "group": "first", "source_image": "source.png"}
            with (
                patch("vinishko.pipeline.steps.vis_searcher.search.fetch_model", return_value=files),
                patch("vinishko.pipeline.steps.vis_searcher.search.inspect", return_value=info),
                patch("vinishko.pipeline.steps.vis_searcher.search.Encoder", return_value=SimpleNamespace(description="stub")),
                patch("vinishko.pipeline.steps.vis_searcher.search.sample_payload", return_value=payload),
            ):
                searcher = VisSearcher(cfg, device=SimpleNamespace(precision="float32"), client=object())
                found = searcher._candidate(crop(), payload, 0.9, True)
            self.assertEqual(searcher.image_field, "source_image")
            self.assertEqual(found.image.shape, (4, 4, 3))

    def test_reference_image_bytes_are_cached_without_reencoding(self) -> None:
        original = b"RIFF\x00\x00\x00\x00WEBPoriginal"

        class ClientStub:
            calls = 0

            def get_object(self, **kwargs: str) -> dict:
                self.calls += 1
                self.last_key = kwargs["Key"]
                return {"Body": BytesIO(original)}

        client = ClientStub()
        with tempfile.TemporaryDirectory() as temp:
            store = S3Store("https://example.invalid", "wine", "data/catalog/images", Path(temp), client=client)
            self.assertEqual(store.get_bytes("label.webp"), original)
            self.assertEqual(store.get_bytes("label.webp"), original)
        self.assertEqual(client.calls, 1)
        self.assertEqual(client.last_key, "data/catalog/images/label.webp")

    def test_pipeline_enables_reranker_with_searcher(self) -> None:
        bottle = crop()
        top = candidate(bottle, "one", "one", ["one"])

        class NormalizerStub:
            def __call__(self, _image: str) -> list[BottleCrop]:
                return [bottle]

        class SearcherStub:
            def __call__(self, _crops: list[BottleCrop]) -> list[BottleCandidates]:
                return [BottleCandidates(bottle, [top])]

        with patch("vinishko.pipeline.steps.near_duplicates.rerank.predict") as model:
            result = Pipeline(normalizer=NormalizerStub(), searcher=SearcherStub())("image")
        self.assertEqual(result.search[0].selection["slug"], "one")
        self.assertIn("rerank", result.timings)
        model.assert_not_called()

    def test_pipeline_can_stop_after_vector_search(self) -> None:
        bottle = crop()
        top = candidate(bottle, "first", "group", ["first", "second"])
        second = candidate(bottle, "second", "group", ["first", "second"])

        class NormalizerStub:
            def __call__(self, _image: str) -> list[BottleCrop]:
                return [bottle]

        class SearcherStub:
            def __call__(self, _crops: list[BottleCrop]) -> list[BottleCandidates]:
                return [BottleCandidates(bottle, [top, second])]

        with patch("vinishko.pipeline.steps.near_duplicates.rerank.predict") as model:
            result = Pipeline(normalizer=NormalizerStub(), searcher=SearcherStub(), enable_rerank=False)("image")
        self.assertEqual([c.slug for c in result.search[0].candidates], ["first", "second"])
        self.assertNotIn("rerank", result.timings)
        model.assert_not_called()

    def test_singleton_returns_top_without_model_or_reference_fetch(self) -> None:
        bottle = crop()
        top = candidate(bottle, "one", "one", ["one"])
        store = FakeStore()
        with patch("vinishko.pipeline.steps.near_duplicates.rerank.predict") as model:
            result = NearDuplicateReranker(reference_store=store)([BottleCandidates(bottle, [top])])[0]
        self.assertEqual([c.slug for c in result.candidates], ["one"])
        self.assertEqual(result.selection["source"], "vector")
        self.assertEqual(store.names, [])
        model.assert_not_called()

    def test_group_uses_all_members_and_returns_selected_slug(self) -> None:
        bottle = crop()
        top = candidate(bottle, "first", "group", ["first", "second"])
        second = candidate(bottle, "second", "group", ["first", "second"])
        outsider = candidate(bottle, "outsider", "other", ["outsider"])
        store = FakeStore()
        with patch("vinishko.pipeline.steps.near_duplicates.rerank.predict", return_value=model_response("second")) as model:
            result = NearDuplicateReranker(reference_store=store)([BottleCandidates(bottle, [top, second, outsider])])[0]
        self.assertEqual([c.slug for c in result.candidates], ["second"])
        self.assertEqual(result.selection["source"], "ndr_v5")
        self.assertEqual(store.names, ["first.webp", "second.webp"])
        request = model.call_args.args[0]
        self.assertEqual([c["slug"] for c in request["group"]["candidates"]], ["first", "second"])
        self.assertEqual(request["group"]["candidates"][0]["grapes"], "Пино нуар")
        self.assertTrue(request["group"]["candidates"][0]["reference_image_bytes"].startswith(b"RIFF"))
        self.assertTrue(request["query"]["image_bytes"].startswith(b"\x89PNG"))

    def test_not_found_becomes_rejected_bottle(self) -> None:
        bottle = crop()
        candidates = [candidate(bottle, slug, "group", ["first", "second"]) for slug in ("first", "second")]
        with patch("vinishko.pipeline.steps.near_duplicates.rerank.predict", return_value=model_response("not_found")):
            result = NearDuplicateReranker()([BottleCandidates(bottle, candidates)])[0]
        self.assertIsInstance(result, UnmatchedBottle)
        self.assertEqual(result.rejected.uuid, bottle.uuid)
        self.assertEqual(str(result.rejected.reason), "near_duplicate_not_found")

    def test_model_contract_error_is_not_replaced_with_top_candidate(self) -> None:
        bottle = crop()
        candidates = [candidate(bottle, slug, "group", ["first", "second"]) for slug in ("first", "second")]
        failure = {"slug": None, "_status": "contract_error", "_error": "invalid JSON", "_trace": {"comparison_calls": []}}
        with tempfile.TemporaryDirectory() as temp, patch("vinishko.pipeline.steps.near_duplicates.rerank.predict", return_value=failure):
            trace_dir = Path(temp)
            with self.assertRaisesRegex(NearDuplicateError, "contract_error"):
                NearDuplicateReranker(trace_dir=trace_dir)([BottleCandidates(bottle, candidates)])
            self.assertTrue((trace_dir / f"{bottle.uuid}.json").is_file())

    def test_incomplete_group_fails_before_model_call(self) -> None:
        bottle = crop()
        top = candidate(bottle, "first", "group", ["first", "second"])
        with patch("vinishko.pipeline.steps.near_duplicates.rerank.predict") as model, self.assertRaises(NearDuplicateError):
            NearDuplicateReranker()([BottleCandidates(bottle, [top])])
        model.assert_not_called()

    def test_v5_predictor_accepts_images_in_memory_with_strict_slug_enum(self) -> None:
        bottle = crop()
        first = candidate(bottle, "first", "group", ["first", "second"])
        second = candidate(bottle, "second", "group", ["first", "second"])
        reranker = NearDuplicateReranker()
        request = {
            "query": {"image_bytes": b"\x89PNG\r\n\x1a\nquery"},
            "group": {"candidates": [
                {"slug": first.slug, "vintage": "2023", "reference_image_bytes": b"\x89PNG\r\n\x1a\nfirst"},
                {"slug": second.slug, "vintage": "", "reference_image_bytes": b"\x89PNG\r\n\x1a\nsecond"},
            ]},
            "prompts": reranker.prompts,
            "runtime": {"model": "test/model", "api_key_env": "NDR_OFFLINE_TEST_KEY", "image_detail": "original"},
        }
        with patch.dict(os.environ, {"NDR_OFFLINE_TEST_KEY": "test-only"}), patch.object(predictor, "call_model", return_value={
            "status": "ok", "error": None, "selected": model_response("second")["_trace"]["comparison_calls"][0]["selected"], "trace": {},
        }) as call:
            response = predictor.predict(request)
        self.assertEqual(response["slug"], "second")
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.kwargs["output_format"]["json_schema"]["schema"]["properties"]["slug"]["enum"], ["first", "second", "not_found"])
        self.assertEqual(len([part for part in call.call_args.kwargs["api_content"] if part["type"] == "image_url"]), 3)
        self.assertTrue(all(part["image"]["path"] is None for part in call.call_args.kwargs["trace_content"] if part["type"] == "image_url"))

    def test_openrouter_proxy_is_used_only_when_configured(self) -> None:
        def response() -> BytesIO:
            result = BytesIO(b'{"choices": []}')
            result.status = 200
            result.headers = {}
            return result

        kwargs = {
            "runtime": {"api_base": "https://openrouter.ai/api/v1", "model": "test/model"},
            "api_key": "test-only",
            "system_prompt": "test",
            "api_content": [],
            "trace_content": [],
            "output_format": {"json_schema": {"schema": {}}},
            "validator": lambda value: value,
        }
        with patch.dict(os.environ, {"openrouter_http_proxy": "http://127.0.0.1:8080"}), patch.object(predictor.urllib.request, "build_opener") as build, patch.object(predictor.urllib.request, "urlopen") as direct:
            build.return_value.open.side_effect = lambda *_args, **_kwargs: response()
            predictor.call_model(**kwargs)
            build.assert_called_once()
            self.assertEqual(build.call_args.args[0].proxies, {"http": "http://127.0.0.1:8080", "https": "http://127.0.0.1:8080"})
            build.return_value.open.assert_called_once()
            direct.assert_not_called()

        with patch.dict(os.environ, {"openrouter_http_proxy": ""}), patch.object(predictor.urllib.request, "build_opener") as build, patch.object(predictor.urllib.request, "urlopen", side_effect=lambda *_args, **_kwargs: response()) as direct:
            predictor.call_model(**kwargs)
            build.assert_not_called()
            direct.assert_called_once()


if __name__ == "__main__":
    unittest.main()
