"""Офлайн-проверки отладочного CLI без загрузки моделей и обращения к сервисам."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from click.testing import CliRunner
from PIL import Image

from vinishko.pipeline import debug
from vinishko.pipeline.structs import BottleCandidates, BottleCrop, Candidate


class DebugCliTests(unittest.TestCase):
    def test_two_dataset_images_stop_after_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "test"
            pictures = dataset / "images"
            pictures.mkdir(parents=True)
            names = ["first.jpg", "second.jpg", "third.jpg"]
            with (dataset / "test.csv").open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["image_filename", "slug"])
                writer.writeheader()
                writer.writerows({"image_filename": name, "slug": ""} for name in names)
            for name in names:
                (pictures / name).write_bytes(b"image bytes are mocked")
            output = root / "runs"
            normalizer = Mock(return_value=[])
            with (
                patch.object(debug, "load_norm_config", return_value={"output": {"format": "jpg"}}),
                patch.object(debug, "Normalizer", return_value=normalizer),
                patch.object(debug, "VisSearcher") as searcher,
                patch.object(debug, "open_image", return_value=Image.new("RGB", (2, 2))),
                patch.object(debug, "write_outputs"),
            ):
                result = CliRunner().invoke(debug.main, [str(dataset), "-o", str(output), "--stage", "normalization", "--limit", "2"])
            self.assertEqual(result.exit_code, 0, str(result.exception))
            self.assertEqual(normalizer.call_count, 2)
            searcher.assert_not_called()
            self.assertTrue((output / "first" / "result.json").is_file())
            self.assertTrue((output / "second" / "result.json").is_file())
            self.assertFalse((output / "third").exists())

    def test_search_uses_saved_normalization_without_loading_sam3(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "photo.jpg"
            Image.new("RGB", (2, 2)).save(image)
            output = root / "runs"
            norm_dir = output / "photo" / "normalization"
            norm_dir.mkdir(parents=True)
            exact = np.full((2, 2, 3), 137, np.uint8)
            crop = BottleCrop(1, 0.9, [], [], 0.0, exact, {"angle_deg": 0.0})
            markup = debug.normalization_markup([crop], image.stem, "jpg")
            marker = norm_dir / "markup.json"
            marker.write_text(json.dumps(markup), encoding="utf-8")
            np.save(norm_dir / markup[0]["pipeline_crop_file"], exact)
            Image.new("RGB", (2, 2), (0, 0, 0)).save(norm_dir / markup[0]["crop_file"])
            (norm_dir / "photo_b1.json").write_text(
                json.dumps({"uuid": crop.uuid, "source": str(image)}), encoding="utf-8"
            )
            (output / "photo" / "result.json").write_text(
                json.dumps({"image": str(image)}), encoding="utf-8"
            )
            old_search = output / "photo" / "search"
            old_search.mkdir()
            (old_search / "old.txt").write_text("stale", encoding="utf-8")
            searcher = Mock(
                side_effect=lambda crops: [
                    BottleCandidates(
                        item,
                        [Candidate("wine", 0.8, exact, item, "wine", True, {})],
                    )
                    for item in crops
                ]
            )
            with (
                patch.dict("os.environ", {"MATCH_MODE": "groups"}),
                patch.object(debug, "load_search_config", return_value=SimpleNamespace(debug_path=None)),
                patch.object(debug, "VisSearcher", return_value=searcher),
                patch.object(debug, "Normalizer") as normalizer,
                patch.object(debug, "open_image") as open_image,
                patch.object(debug, "load_norm_config") as load_norm_config,
            ):
                result = CliRunner().invoke(
                    debug.main,
                    [str(image), "-o", str(output), "--stage", "search", "--skip-normalization"],
                )
            self.assertEqual(result.exit_code, 0, str(result.exception))
            normalizer.assert_not_called()
            open_image.assert_not_called()
            load_norm_config.assert_not_called()
            self.assertEqual(marker.read_text(encoding="utf-8"), json.dumps(markup))
            self.assertFalse((old_search / "old.txt").exists())
            passed = searcher.call_args.args[0][0]
            self.assertEqual(passed.uuid, crop.uuid)
            np.testing.assert_array_equal(passed.crop, exact)
            report = json.loads((output / "photo" / "result.json").read_text(encoding="utf-8"))
            self.assertIn("normalization_cached", report["timings_s"])
            self.assertEqual(report["bottles"][0]["candidates"][0]["slug"], "wine")

    def test_skip_normalization_requires_later_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "photo.jpg"
            image.write_bytes(b"image")
            result = CliRunner().invoke(
                debug.main,
                [str(image), "-o", str(Path(temp) / "runs"), "--stage", "normalization", "--skip-normalization"],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("--stage search", result.output)

    def test_raw_mode_searches_whole_image_without_cached_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "photo.png"
            Image.new("RGB", (4, 3), (21, 42, 63)).save(image)
            output = root / "runs_raw"
            searcher = Mock(
                side_effect=lambda crops: [
                    BottleCandidates(
                        item,
                        [Candidate("wine", 0.8, item.crop, item, "wine", True, {})],
                    )
                    for item in crops
                ]
            )
            reranker = Mock(side_effect=lambda results: results)
            with (
                patch.dict("os.environ", {"MATCH_MODE": "groups"}),
                patch.object(debug, "load_search_config", return_value=SimpleNamespace(debug_path=None)),
                patch.object(debug, "VisSearcher", return_value=searcher),
                patch.object(debug.NearDuplicateReranker, "from_search_config", return_value=reranker),
                patch.object(debug, "Normalizer") as normalizer,
                patch.object(debug, "load_norm_config") as load_norm_config,
            ):
                result = CliRunner().invoke(
                    debug.main,
                    [str(image), "-o", str(output), "--stage", "search", "--no-normalization"],
                )
                full = CliRunner().invoke(
                    debug.main,
                    [str(image), "-o", str(output), "--stage", "full", "--no-normalization"],
                )
            self.assertEqual(result.exit_code, 0, str(result.exception))
            self.assertEqual(full.exit_code, 0, str(full.exception))
            normalizer.assert_not_called()
            load_norm_config.assert_not_called()
            reranker.assert_called_once()
            passed = searcher.call_args.args[0][0]
            self.assertEqual(passed.crop.shape, (3, 4, 3))
            np.testing.assert_array_equal(passed.crop[0, 0], [21, 42, 63])
            self.assertEqual(len(searcher.call_args.args[0]), 1)
            self.assertFalse((output / "photo" / "normalization").exists())
            report = json.loads((output / "photo" / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(report["input_mode"], "raw")
            self.assertIn("raw_input", report["timings_s"])

    def test_bypass_modes_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "photo.jpg"
            image.write_bytes(b"image")
            result = CliRunner().invoke(
                debug.main,
                [str(image), "-o", str(Path(temp) / "runs"), "--stage", "search", "--skip-normalization", "--no-normalization"],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("--skip-normalization", result.output)

    def test_old_jpeg_cache_and_rejection_are_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "photo.jpg"
            image.write_bytes(b"source file")
            norm_dir = root / "runs" / "photo" / "normalization"
            norm_dir.mkdir(parents=True)
            crop = BottleCrop(1, 0.9, [], [], 0.0, np.zeros((2, 2, 3), np.uint8), {})
            row = debug.normalization_markup([crop], image.stem, "jpg")[0]
            row.pop("pipeline_crop_file")
            row.pop("crop_info")
            rows = [row, {"status": "rejected", "reason": "not_target", "detail": "score low", "score": 0.1, "bottle": [], "label": None, "uuid": "rejected"}]
            (norm_dir / "markup.json").write_text(json.dumps(rows), encoding="utf-8")
            Image.new("RGB", (2, 2), (23, 23, 23)).save(norm_dir / row["crop_file"])
            (norm_dir / "photo_b1.json").write_text(
                json.dumps({"uuid": crop.uuid, "source": str(image), "bottle_angle": 0.0, "bottle_crop_matrix": [[1, 0, 0], [0, 1, 0]]}),
                encoding="utf-8",
            )

            loaded = debug.load_normalization(image, root / "runs")

            self.assertEqual([item.uuid for item in loaded], [crop.uuid, "rejected"])
            self.assertEqual(loaded[0].crop.shape, (2, 2, 3))
            self.assertEqual(loaded[1].reason, "not_target")


if __name__ == "__main__":
    unittest.main()
