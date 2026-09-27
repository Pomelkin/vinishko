"""Офлайн-проверка отдельного VLM-шага и клиента SDK."""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from vinishko.pipeline.pipeline import Pipeline
from vinishko.pipeline.steps.vanilla_vlm_rerank import VanillaVlmReranker
from vinishko.pipeline.steps.vanilla_vlm_rerank.config import VanillaVlmSettings
from vinishko.pipeline.steps.vis_searcher.configs import load_config
from vinishko.pipeline.structs import BottleCandidates, BottleCrop, Candidate


class VanillaVlmTests(unittest.TestCase):
    def test_sdk_receives_images_slugs_and_structured_schema(self) -> None:
        crop = BottleCrop(1, 0.9, [], [], 0.0, np.zeros((4, 4, 3), dtype=np.uint8), {})
        candidates = [Candidate(slug, 0.9, crop.crop, crop, "unused", True, {"slug": slug}) for slug in ("first", "second", "third")]
        calls = []

        def parse(**kwargs):
            calls.append(kwargs)
            parsed = kwargs["response_format"].model_validate({
                "slug": "third",
                "checklist": {
                    "maker": {"observation": "совпадает"},
                    "profile": {"observation": "совпадает"},
                    "year": {"observation": "не виден"},
                },
            })
            return SimpleNamespace(id="offline", choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
        reranker = VanillaVlmReranker(client=client)
        result = reranker([BottleCandidates(crop, candidates)])[0]
        self.assertEqual([item.slug for item in result.candidates], ["third"])
        self.assertEqual(result.selection["source"], "vanilla_vlm")
        self.assertEqual(len(calls), 1)
        schema = calls[0]["response_format"].model_json_schema()
        self.assertEqual(schema["properties"]["slug"]["enum"], ["first", "second", "third", "not_found"])
        content = calls[0]["messages"][1]["content"]
        self.assertEqual(sum(part["type"] == "image_url" for part in content), 4)
        self.assertIn('"slug": "first"', content[2]["text"])
        self.assertEqual(calls[0]["max_completion_tokens"], reranker.settings.max_completion_tokens)
        self.assertEqual(calls[0]["extra_body"], reranker.settings.extra_body)
        for name in ("resolve_multiple_same", "compare_year_matters", "compare_year_not_matter"):
            source = Path(__file__).parents[1] / "near_duplicates" / "prompts" / f"{name}.txt"
            copied = Path(__file__).parent / "prompts" / f"{name}.txt"
            self.assertEqual(source.read_bytes(), copied.read_bytes())

    def test_mode_and_sdk_proxy(self) -> None:
        cfg = load_config()
        cfg.reference_images = None
        cfg.catalog_csv = None
        with patch.dict(os.environ, {"MATCH_MODE": "vanilla_vlm", "OPENROUTER_API_KEY": "offline", "openrouter_http_proxy": "http://127.0.0.1:8080"}):
            pipeline = Pipeline(normalizer=object(), searcher=SimpleNamespace(cfg=cfg))
        self.assertEqual(cfg.top_k, 3)
        self.assertEqual(cfg.search.mode, "top_n")
        self.assertIsInstance(pipeline.reranker, VanillaVlmReranker)

        with patch("vinishko.pipeline.steps.vanilla_vlm_rerank.rerank.httpx2.Client") as http_client, patch("vinishko.pipeline.steps.vanilla_vlm_rerank.rerank.OpenAI") as sdk:
            pipeline.reranker._client()
        http_client.assert_called_once_with(proxy="http://127.0.0.1:8080", trust_env=False)
        sdk.assert_called_once()
        self.assertEqual(sdk.call_args.kwargs["base_url"], "https://openrouter.ai/api/v1")

    def test_settings_read_environment(self) -> None:
        with patch.dict(os.environ, {
            "OPENROUTER_MODEL": "offline/model",
            "VANILLA_VLM_MAX_COMPLETION_TOKENS": "77",
            "VANILLA_VLM_EXTRA_BODY": '{"provider":{"only":["offline"]}}',
        }):
            settings = VanillaVlmSettings(_env_file=None)
        self.assertEqual(settings.model, "offline/model")
        self.assertEqual(settings.max_completion_tokens, 77)
        self.assertEqual(settings.extra_body, {"provider": {"only": ["offline"]}})


if __name__ == "__main__":
    unittest.main()
