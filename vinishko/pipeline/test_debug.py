"""Офлайн-проверки отладочного CLI без загрузки моделей и обращения к сервисам."""

import csv
import json
import tempfile
import threading
import time
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
    def test_indexed_slugs_reads_all_qdrant_pages(self) -> None:
        client = Mock()
        client.scroll.side_effect = [
            ([SimpleNamespace(payload={"slug": "first"})], 123),
            ([SimpleNamespace(payload={"slug": "second"})], None),
        ]
        self.assertEqual(debug.indexed_slugs(client, "catalog"), {"first", "second"})
        self.assertEqual(client.scroll.call_args.kwargs["offset"], 123)

    def test_full_run_saves_search_candidates_before_ndr(self) -> None:
        rgb = np.zeros((2, 2, 3), np.uint8)
        crop = BottleCrop(1, 0.9, [], [], 0.0, rgb, {}, rgb, {})
        first = Candidate("first", 0.9, rgb, crop, "group", True, {})
        second = Candidate("second", 0.9, rgb, crop, "group", False, {})
        searcher = Mock(side_effect=lambda crops: [BottleCandidates(crops[0], [first, second])])
        reranker = Mock(side_effect=lambda rows: [BottleCandidates(rows[0].crop, [second])])
        pipeline = debug.Pipeline(Mock(return_value=[crop]), searcher, reranker)
        with tempfile.TemporaryDirectory() as temp, patch.object(debug, "write_debug_result") as written:
            debug.run_debug_batches([Path(temp) / "photo.jpg"], Path(temp) / "runs", pipeline, searcher, reranker, None, "precomputed", 1)
        saved = searcher.dump.call_args.args[0][0]
        final = written.call_args.args[3].search[0]
        self.assertEqual([candidate.slug for candidate in saved.candidates], ["first", "second"])
        self.assertEqual([candidate.slug for candidate in final.candidates], ["second"])

    def test_metrics_separate_indexed_search_recall_from_ndr_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "runs"
            cases = [
                ("positive", "wanted", [{"slug": "neighbor", "group": "group", "score": 0.9, "group_slugs": ["neighbor", "wanted"]}, {"slug": "wanted", "group": "group", "score": 0.9}], "wanted"),
                ("unindexed", "absent", [{"slug": "neighbor", "group": "group", "score": 0.7}], "neighbor"),
                ("empty", "", [], ""),
            ]
            images = []
            truth = {}
            for stem, expected, candidates, final in cases:
                image = root / f"{stem}.jpg"
                images.append(image)
                truth[image.name] = expected
                folder = output / stem
                (folder / "search").mkdir(parents=True)
                (folder / "search" / "results.json").write_text(json.dumps({"queries": [{"query": stem, "rejected": None if candidates else {"reason": "no_match"}, "selection": None, "candidates": candidates}]}), encoding="utf-8")
                bottle = {"uuid": stem, "status": "candidates", "candidates": [{"slug": final}]} if final else {"uuid": stem, "status": "rejected", "step": "rerank"}
                (folder / "result.json").write_text(json.dumps({"bottles": [bottle], "timings_s": {"search": 1.0}}), encoding="utf-8")
            catalog = root / "catalog.csv"
            catalog.write_text("Slug,near_duplicate_group_slug\nwanted,group\nneighbor,group\nabsent,other\n", encoding="utf-8")
            result = debug.write_metrics(images, output, "metrics", truth, catalog, known={"wanted", "neighbor"})
            self.assertEqual(result["search"]["with_answer"], 1)
            self.assertEqual(result["search"]["answer_not_indexed"], 1)
            self.assertEqual(result["search"]["recall@1"], 0.0)
            self.assertEqual(result["search"]["recall@3"], 1.0)
            self.assertEqual(result["search"]["group_recall"], 1.0)
            self.assertEqual(result["search"]["correct_reject"], 1.0)
            self.assertEqual(result["ndr"]["accuracy"], 0.6667)
            self.assertEqual(result["ndr"]["accuracy_indexed_or_empty"], 1.0)
            self.assertEqual(result["ndr"]["correct_slug_rate_for_indexed_answers"], 1.0)
            self.assertNotIn("exact_match", result["ndr"])
            self.assertNotIn("recall@3", result["ndr"])

    def test_full_batches_overlap_ndr_across_search_batches(self) -> None:
        lock = threading.Lock()
        active = peak = 0

        class SlowReranker:
            trace_dir = None

            def __call__(self, rows: list[BottleCandidates]) -> list[BottleCandidates]:
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.05)
                with lock:
                    active -= 1
                return rows

        def normalize(_image: Path) -> list[BottleCrop]:
            rgb = np.zeros((2, 2, 3), np.uint8)
            return [BottleCrop(1, 0.9, [], [], 0.0, rgb, {}, rgb, {})]

        def search(crops: list[BottleCrop]) -> list[BottleCandidates]:
            return [BottleCandidates(crop, [Candidate("wine", 0.9, crop.box_crop, crop, "wine", True, {})]) for crop in crops]

        with tempfile.TemporaryDirectory() as temp, patch.object(debug, "write_debug_result") as written:
            root = Path(temp)
            images = [root / f"{index}.jpg" for index in range(3)]
            reranker = SlowReranker()
            pipeline = debug.Pipeline(Mock(side_effect=normalize), Mock(side_effect=search), reranker)
            debug.run_debug_batches(images, root / "runs", pipeline, None, reranker, None, "precomputed", 1)
        self.assertEqual(written.call_count, 3)
        self.assertGreater(peak, 1)

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
            self.assertEqual(metrics["search"]["answer_not_indexed"], 1)
            self.assertIsNone(metrics["search"]["recall@1"])
            self.assertEqual(metrics["search"]["not_indexed_accepted"], 0.0)

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
                patch.object(debug, "known_slugs_for_metrics", return_value={"first"}),
                patch.object(debug, "Normalizer") as sam3,
            ):
                result = CliRunner().invoke(debug.main, [str(dataset), "-o", str(output), "--stage", "search", "--normalized-dir", str(normalized), "--limit", "2"])
            self.assertEqual(result.exit_code, 0, str(result.exception))
            self.assertNotIn("нет готового кропа", result.output)
            sam3.assert_not_called()
            self.assertEqual(len(seen), 2)
            self.assertEqual(seen[0].crop.shape, (2, 3, 3))
            self.assertEqual(seen[0].box_crop.shape, (8, 12, 3))
            np.testing.assert_array_equal(seen[0].box_crop, np.asarray(debug.open_image(originals / "first.webp")))
            self.assertEqual(debug.PrecomputedNormalizer(normalized)(originals / "first.webp")[0].box_crop.shape, (8, 12, 3))
            self.assertFalse((output / "missing").exists())
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["images"], 2)
            self.assertEqual(metrics["search"]["images"], 2)
            self.assertEqual(metrics["search"]["with_answer"], 1)
            self.assertEqual(metrics["search"]["recall@1"], 0.0)
            self.assertEqual(metrics["search"]["without_answer"], 1)
            self.assertEqual(metrics["search"]["correct_reject"], 1.0)

            with (
                patch.object(debug, "load_search_config", return_value=SimpleNamespace(debug_path=None, catalog_csv=catalog)),
                patch.object(debug, "VisSearcher", return_value=Mock(side_effect=search)),
                patch.object(debug, "known_slugs_for_metrics", return_value={"first"}),
            ):
                full = CliRunner().invoke(debug.main, [str(dataset), "-o", str(output), "--stage", "search", "--normalized-dir", str(normalized)])
            self.assertEqual(full.exit_code, 0, str(full.exception))
            complete_metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(complete_metrics["images"], 3)
            self.assertEqual(complete_metrics["searched_images"], 2)
            self.assertEqual(complete_metrics["missing_normalization"], 1)
            self.assertEqual(complete_metrics["search"]["no_bottle"], 0.3333)

    def test_paired_s3_crops_work_with_skip_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "test"
            pictures = dataset / "images"
            pictures.mkdir(parents=True)
            source = pictures / "first.webp"
            Image.new("RGB", (12, 8), (20, 30, 40)).save(source)
            with (dataset / "test.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["image_filename", "slug"])
                writer.writeheader()
                writer.writerow({"image_filename": source.name, "slug": "wine"})
            normalized = root / "normalized"
            crop_dir = normalized / "images_crop"
            box_dir = normalized / "images_crop_box"
            crop_dir.mkdir(parents=True)
            box_dir.mkdir()
            Image.new("RGB", (3, 2), (201, 22, 33)).save(crop_dir / "first.jpg")
            Image.new("RGB", (7, 5), (30, 190, 60)).save(box_dir / "first.jpg")
            seen = []

            def search(crops: list[BottleCrop]) -> list[BottleCandidates]:
                seen.extend(crops)
                return [BottleCandidates(crop, [Candidate("wine", 0.9, crop.box_crop, crop, "wine", True, {})]) for crop in crops]

            output = root / "runs"
            catalog = root / "catalog.csv"
            catalog.write_text("Slug\nwine\n", encoding="utf-8")
            with (
                patch.object(debug, "load_search_config", return_value=SimpleNamespace(debug_path=None, catalog_csv=catalog)),
                patch.object(debug, "VisSearcher", return_value=Mock(side_effect=search)),
                patch.object(debug, "known_slugs_for_metrics", return_value={"wine"}),
                patch.object(debug, "Normalizer") as sam3,
            ):
                result = CliRunner().invoke(debug.main, [str(dataset), "-o", str(output), "--stage", "search", "--skip-normalization", "--normalized-dir", str(normalized)])
            self.assertEqual(result.exit_code, 0, str(result.exception))
            sam3.assert_not_called()
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0].crop.shape, (2, 3, 3))
            self.assertEqual(seen[0].box_crop.shape, (5, 7, 3))
            np.testing.assert_array_equal(seen[0].box_crop, seen[0].original)
            self.assertEqual(seen[0].box_info["mode"], "precomputed")
            report = json.loads((output / "first" / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(report["input_mode"], "precomputed")
            found = BottleCandidates(seen[0], [Candidate("wine", 0.9, seen[0].box_crop, seen[0], "wine", True, {})])
            debug.VisSearcher.dump(SimpleNamespace(cfg=SimpleNamespace(search=SimpleNamespace(mode="groups"), top_k=1)), [found], output / "first" / "search")
            _, restored = debug.load_saved_search(source, output, normalized)
            np.testing.assert_array_equal(restored[0].crop.original, seen[0].box_crop)

    def test_ndr_uses_saved_search_without_repeating_search(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "test"
            pictures = dataset / "images"
            pictures.mkdir(parents=True)
            normalized = root / "normalized"
            normalized.mkdir()
            image = pictures / "first.jpg"
            missing = pictures / "missing.jpg"
            for path in (image, missing):
                Image.new("RGB", (12, 8), (20, 30, 40)).save(path)
            Image.new("RGB", (3, 2)).save(normalized / "first.jpg")
            with (dataset / "test.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["image_filename", "slug"])
                writer.writeheader()
                writer.writerows([{"image_filename": image.name, "slug": "second"}, {"image_filename": missing.name, "slug": ""}])
            catalog = root / "catalog.csv"
            catalog.write_text("Slug\nfirst\nsecond\n", encoding="utf-8")
            output = root / "runs"
            out = output / image.stem
            out.mkdir(parents=True)
            rgb = np.full((4, 5, 3), 90, dtype=np.uint8)
            box = np.full((8, 12, 3), 180, dtype=np.uint8)
            crop = BottleCrop(1, 0.8, [], [], 0.0, rgb, {}, box, {})
            candidates = [Candidate(slug, 0.9, rgb, crop, "group", rank == 0, {}) for rank, slug in enumerate(("first", "second"))]
            found = BottleCandidates(crop, candidates)
            debug.VisSearcher.dump(SimpleNamespace(cfg=SimpleNamespace(search=SimpleNamespace(mode="groups"), top_k=1)), [found], out / "search")
            report = debug.summary(image, debug.PipelineResult([crop], [found], {"search": 1.25}))
            report["input_mode"] = "precomputed"
            (out / "result.json").write_text(json.dumps(report), encoding="utf-8")
            saved_search = (out / "search" / "results.json").read_bytes()
            reranker = Mock(side_effect=lambda rows: [BottleCandidates(rows[0].crop, [rows[0].candidates[1]], {"source": "ndr_v5", "slug": "second"})])
            with (
                patch.object(debug, "load_search_config", return_value=SimpleNamespace(catalog_csv=catalog)),
                patch.object(debug.NearDuplicateReranker, "from_search_config", return_value=reranker),
                patch.object(debug, "known_slugs_for_metrics", return_value={"first", "second"}),
                patch.object(debug, "VisSearcher") as searcher,
                patch.object(debug, "Normalizer") as normalizer,
            ):
                result = CliRunner().invoke(debug.main, [str(dataset), "-o", str(output), "--stage", "ndr", "--normalized-dir", str(normalized)])
            self.assertEqual(result.exit_code, 0, str(result.exception))
            searcher.assert_not_called()
            normalizer.assert_not_called()
            self.assertEqual((out / "search" / "results.json").read_bytes(), saved_search)
            passed = reranker.call_args.args[0][0]
            self.assertEqual(passed.crop.uuid, crop.uuid)
            self.assertEqual(passed.crop.box_crop.shape, box.shape)
            self.assertEqual(passed.crop.original.shape, (8, 12, 3))
            np.testing.assert_array_equal(passed.crop.original[0, 0], [20, 30, 40])
            self.assertEqual(passed.candidates[0].payload["group_slugs"], ["first", "second"])
            final = json.loads((out / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(final["bottles"][0]["candidates"][0]["slug"], "second")
            self.assertEqual(final["input_mode"], "precomputed")
            self.assertIn("rerank", final["timings_s"])
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["stage"], "ndr")
            self.assertEqual(metrics["missing_normalization"], 1)
            self.assertEqual(metrics["search"]["recall@1"], 0.0)
            self.assertEqual(metrics["search"]["recall@3"], 1.0)
            self.assertEqual(metrics["ndr"]["accuracy"], 1.0)
            self.assertNotIn("recall@1", metrics["ndr"])

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
