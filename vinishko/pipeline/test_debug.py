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
from vinishko.pipeline.steps.vis_searcher.search import SearchReason
from vinishko.pipeline.structs import BottleCandidates, BottleCrop, Candidate, UnmatchedBottle


class DebugCliTests(unittest.TestCase):
    def test_slug_absent_from_catalog_is_scored_as_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "outside.jpg"
            image.write_bytes(b"unused")
            output = root / "runs"
            result_dir = output / image.stem
            result_dir.mkdir(parents=True)
            (result_dir / "result.json").write_text(json.dumps({"bottles": [{"status": "rejected"}], "timings_s": {"search": 1.0}}), encoding="utf-8")
            catalog = root / "catalog.csv"
            catalog.write_text("Slug\ninside\n", encoding="utf-8")
            metrics = debug.write_metrics([image], output, "search", {image.name: "outside"}, catalog)
            self.assertEqual(metrics["all"]["accuracy_at_1"], 1.0)
            self.assertEqual(metrics["catalog_only"]["images"], 0)
            self.assertEqual(metrics["not_found"], 1)

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
            crop = BottleCrop(1, 0.9, [], [], 0.0, exact, {"angle_deg": 0.0}, exact, {})
            markup = debug.normalization_markup([crop], image.stem, "jpg")
            marker = norm_dir / "markup.json"
            marker.write_text(json.dumps(markup), encoding="utf-8")
            np.save(norm_dir / markup[0]["pipeline_crop_file"], exact)
            np.save(norm_dir / markup[0]["pipeline_box_file"], exact)
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

    def test_s3_crops_are_used_for_search_and_metrics_without_sam3(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "test"
            originals = dataset / "images"
            originals.mkdir(parents=True)
            normalized = root / "normalized"
            normalized.mkdir()
            for name in ("first.webp", "missing.webp", "third.webp"):
                Image.new("RGB", (12, 8), (12, 34, 56)).save(originals / name)
            for stem in ("first", "third"):
                Image.new("RGB", (3, 2), (201, 22, 33)).save(normalized / f"{stem}.jpg", quality=100)
            with (dataset / "test.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["image_filename", "slug"])
                writer.writeheader()
                writer.writerows([{"image_filename": "first.webp", "slug": "first"}, {"image_filename": "third.webp", "slug": "not_found"}, {"image_filename": "missing.webp", "slug": ""}])
            catalog = root / "catalog.csv"
            catalog.write_text("Slug\nfirst\n", encoding="utf-8")
            seen = []

            def search(crops: list[BottleCrop]) -> list[BottleCandidates | UnmatchedBottle]:
                seen.extend(crops)
                first, third = crops
                return [
                    BottleCandidates(first, [Candidate("other", 0.9, first.box_crop, first, "other", True, {})]),
                    UnmatchedBottle(third, third.reject(SearchReason.NO_MATCH, "no match")),
                ]

            output = root / "runs"
            with (
                patch.object(debug, "load_search_config", return_value=SimpleNamespace(debug_path=None, catalog_csv=catalog)),
                patch.object(debug, "VisSearcher", return_value=Mock(side_effect=search)),
                patch.object(debug, "Normalizer") as sam3,
            ):
                result = CliRunner().invoke(debug.main, [str(dataset), "-o", str(output), "--stage", "search", "--normalized-dir", str(normalized), "--limit", "2"])
            self.assertEqual(result.exit_code, 0, str(result.exception))
            self.assertNotIn("нет готового кропа", result.output)
            sam3.assert_not_called()
            self.assertEqual(len(seen), 2)
            self.assertEqual(seen[0].crop.shape, (2, 3, 3))
            self.assertEqual(seen[0].box_crop.shape, (2, 3, 3))
            self.assertEqual(debug.PrecomputedNormalizer(normalized)(originals / "first.webp")[0].box_crop.shape, (8, 12, 3))
            self.assertFalse((output / "missing").exists())
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["images"], 2)
            self.assertEqual(metrics["all"]["images"], 2)
            self.assertEqual(metrics["all"]["accuracy_at_1"], 0.5)
            self.assertEqual(metrics["catalog_only"]["images"], 1)
            self.assertEqual(metrics["catalog_only"]["accuracy_at_1"], 0.0)
            self.assertEqual(metrics["not_found"], 1)
            self.assertEqual(metrics["correct_reject"], 1.0)

            with (
                patch.object(debug, "load_search_config", return_value=SimpleNamespace(debug_path=None, catalog_csv=catalog)),
                patch.object(debug, "VisSearcher", return_value=Mock(side_effect=search)),
            ):
                full = CliRunner().invoke(debug.main, [str(dataset), "-o", str(output), "--stage", "search", "--normalized-dir", str(normalized)])
            self.assertEqual(full.exit_code, 0, str(full.exception))
            complete_metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(complete_metrics["images"], 3)
            self.assertEqual(complete_metrics["searched_images"], 2)
            self.assertEqual(complete_metrics["missing_normalization"], 1)
            self.assertEqual(complete_metrics["all"]["accuracy_at_1"], 0.6667)

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
            rgb = np.zeros((2, 2, 3), np.uint8)
            crop = BottleCrop(1, 0.9, [], [], 0.0, rgb, {}, rgb, {})
            row = debug.normalization_markup([crop], image.stem, "jpg")[0]
            row.pop("pipeline_crop_file")
            row.pop("crop_info")
            row.pop("box_file")
            row.pop("pipeline_box_file")
            row.pop("box_info")
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
