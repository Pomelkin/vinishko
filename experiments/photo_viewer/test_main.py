import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.photo_viewer import main


class ImageGroupsTest(unittest.TestCase):
    def test_split_is_balanced_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            photo_dir = Path(directory)
            for name in ["a.jpg", "b.png", "c.webp", "d.jpeg", "ignored.txt"]:
                (photo_dir / name).touch()

            with patch.object(main, "PHOTO_DIR", photo_dir):
                main.image_groups.cache_clear()
                first = main.image_groups()
                main.image_groups.cache_clear()

                self.assertEqual(first, main.image_groups())
                self.assertEqual(sorted(map(len, first)), [1, 1, 2])
                self.assertTrue(all(path.suffix != ".txt" for group in first for path in group))


if __name__ == "__main__":
    unittest.main()
