"""Офлайн-проверки группировки кропов и пакетного запроса Qdrant."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from vinishko.pipeline.pipeline import Pipeline
from vinishko.pipeline.steps.vis_searcher.catalog import search_many
from vinishko.pipeline.steps.vis_searcher.device import Device
from vinishko.pipeline.steps.vis_searcher.model import Encoder, Preprocess
from vinishko.pipeline.structs import BottleCandidates, BottleCrop, Candidate


def crop(index: int) -> BottleCrop:
    rgb = np.zeros((2, 2, 3), np.uint8)
    return BottleCrop(index, 0.9, [], [], 0.0, rgb, {}, rgb, {})


class BatchTests(unittest.TestCase):
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
        files = SimpleNamespace(input_size=(2, 2), pad_color=(0, 0, 0), resize="pad", embed_dim=2)
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
