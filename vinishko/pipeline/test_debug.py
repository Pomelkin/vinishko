"""Офлайн-проверки отладочного CLI без загрузки моделей и обращения к сервисам."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner
from PIL import Image

from vinishko.pipeline import debug


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


if __name__ == "__main__":
    unittest.main()
