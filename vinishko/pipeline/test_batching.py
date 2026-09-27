"""Офлайн-проверки группировки кропов и пакетного запроса Qdrant."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image

from vinishko.pipeline.pipeline import Pipeline
from vinishko.pipeline.steps.normalization.normalize import Normalizer
from vinishko.pipeline.steps.vis_searcher.catalog import search_many
from vinishko.pipeline.steps.vis_searcher.device import Device
from vinishko.pipeline.steps.vis_searcher.model import Encoder, Preprocess
from vinishko.pipeline.steps.vis_searcher.search import VisSearcher
from vinishko.pipeline.structs import BottleCandidates, BottleCrop, Candidate


def crop(index: int) -> BottleCrop:
    rgb = np.zeros((2, 2, 3), np.uint8)
    return BottleCrop(index, 0.9, [], [], 0.0, rgb, {}, rgb, {})


class BatchTests(unittest.TestCase):
    def test_normalizer_keeps_original_separate_from_encoder_crop(self) -> None:
        normalizer = object.__new__(Normalizer)
        normalizer.cfg = {"orientation": {}, "fallback": {"min_aspect": 2}, "selection": {"min_bottle_px": 1}}
        normalizer.check_label = Mock(return_value=[[1, 1]])
        normalizer.refine_label = Mock(return_value=[[1, 1]])
        original = np.full((8, 9, 3), 222, np.uint8)
        normalized = np.full((3, 4, 3), 11, np.uint8)
        box = np.full((5, 6, 3), 55, np.uint8)
        with (
            patch("vinishko.pipeline.steps.normalization.normalize.bottle_axis", return_value={"length": 10, "body_width": 2, "angle": 0, "center": (0, 0)}),
            patch("vinishko.pipeline.steps.normalization.normalize.rotated_extent", return_value=(None, (0, 0, 2, 10))),
            patch("vinishko.pipeline.steps.normalization.normalize.render_bottle", return_value=(normalized, {})),
            patch("vinishko.pipeline.steps.normalization.normalize.render_bottle_box", return_value=(box, {})),
        ):
            result = normalizer.judge(Image.fromarray(original), original, original, {"polys": []}, 0.9, 1, 1)
        self.assertIs(result.crop, normalized)
        self.assertIs(result.box_crop, box)
        self.assertIs(result.original, original)

    def test_dino_receives_normalized_crop_not_original_query(self) -> None:
        normalized = np.full((3, 4, 3), 11, np.uint8)
        original = np.full((8, 9, 3), 222, np.uint8)
        bottle = BottleCrop(1, 0.9, [], [], 0.0, normalized, {}, original, {}, original=original)
        searcher = object.__new__(VisSearcher)
        searcher.encoder = Mock(return_value=np.ones((1, 2), np.float32))
        searcher.client = object()
        searcher.collection = "catalog"
        searcher.cfg = SimpleNamespace(top_k=5, debug_path=None, encoder_input="crop")
        searcher._search = Mock(return_value="found")
        with patch("vinishko.pipeline.steps.vis_searcher.search.search_many", return_value=[[]]):
            self.assertEqual(searcher([bottle]), ["found"])
        sent = searcher.encoder.call_args.args[0][0]
        self.assertIs(sent, bottle.crop)
        self.assertIsNot(sent, bottle.original)
        searcher.cfg.encoder_input = "box_crop"
        with patch("vinishko.pipeline.steps.vis_searcher.search.search_many", return_value=[[]]):
            searcher([bottle])
        self.assertIs(searcher.encoder.call_args.args[0][0], bottle.box_crop)

    def test_encoder_preprocess_respects_model_resize_contract(self) -> None:
        image = np.full((2, 4, 3), (200, 0, 0), np.uint8)
        padded = Preprocess((4, 4), (1, 2, 3), "pad")(image)
        squashed = Preprocess((4, 4), (1, 2, 3), "squash")(image)
        np.testing.assert_array_equal(padded[:, 0, 0], [1, 2, 3])
        np.testing.assert_array_equal(squashed[:, 0, 0], [200, 0, 0])

    def test_pipeline_batches_search_and_keeps_image_boundaries(self) -> None:
        first, second, third = crop(1), crop(2), crop(1)
        normalizer = Mock(side_effect=[[first, second], [], [third]])
        searcher = Mock(
            side_effect=lambda crops: [
                BottleCandidates(
                    item,
                    [
                        Candidate(
                            "wine",
                            0.9,
                            np.zeros((2, 2, 3), np.uint8),
                            item,
                            "wine",
                            True,
                            {},
                        )
                    ],
                )
                for item in crops
            ]
        )
        reranker = Mock(side_effect=lambda results: results)
        before_rerank = Mock()
        on_result = Mock()
        pipeline = Pipeline(normalizer, searcher, reranker)

        results = pipeline.run_many(
            ["one", "empty", "three"], before_rerank=before_rerank, on_result=on_result
        )

        searcher.assert_called_once_with([first, second, third])
        self.assertEqual([len(result.search) for result in results], [2, 0, 1])
        self.assertEqual([r.crop for r in results[0].search], [first, second])
        self.assertEqual([r.crop for r in results[2].search], [third])
        self.assertEqual(
            [call.args[0] for call in before_rerank.call_args_list], [0, 2]
        )
        self.assertEqual(reranker.call_count, 2)
        self.assertEqual([call.args[0] for call in on_result.call_args_list], [0, 1, 2])
        self.assertAlmostEqual(
            results[0].timings["search"], 2 * results[2].timings["search"]
        )

    def test_qdrant_queries_share_one_request(self) -> None:
        client = Mock()
        client.query_batch_points.return_value = [
            SimpleNamespace(points=["first"]),
            SimpleNamespace(points=["second"]),
        ]
        vectors = np.array([[0.1, 0.2], [0.3, 0.4]], np.float32)

        found = search_many(client, "catalog", vectors, 1)

        self.assertEqual(found, [["first"], ["second"]])
        client.query_batch_points.assert_called_once()
        requests = client.query_batch_points.call_args.kwargs["requests"]
        self.assertEqual(
            [request.query for request in requests],
            [vectors[0].tolist(), vectors[1].tolist()],
        )
        self.assertTrue(
            all(request.limit == 1 and request.with_payload for request in requests)
        )

    def test_cpu_encoder_limits_openvino_batch(self) -> None:
        files = SimpleNamespace(input_size=(2, 2), pad_color=(0, 0, 0), resize="pad", interpolation="area_cubic", embed_dim=2)
        device = Device(torch.device("cpu"), "openvino", "fp32")
        runner = Mock(side_effect=lambda batch: np.ones((len(batch), 2), np.float32))
        with (
            patch(
                "vinishko.pipeline.steps.vis_searcher.model.download_onnx",
                return_value=Path("model.onnx"),
            ),
            patch(
                "vinishko.pipeline.steps.vis_searcher.model.OpenVinoRunner",
                return_value=runner,
            ),
        ):
            encoder = Encoder(files, device, 16, Path(), cpu_max_batch=2)
            output = encoder([np.zeros((2, 2, 3), np.uint8) for _ in range(3)])

        self.assertEqual(output.shape, (3, 2))
        self.assertEqual(
            [call.args[0].shape[0] for call in runner.call_args_list], [1, 2, 1]
        )


if __name__ == "__main__":
    unittest.main()
