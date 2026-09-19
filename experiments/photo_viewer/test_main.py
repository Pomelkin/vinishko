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
            catalog_csv = photo_dir / "catalog.csv"
            catalog_csv.write_text(
                "Slug,Название фото,Название вина\n"
                "slug-a,a.jpg,Wine A\nslug-b,b.png,Wine B\nslug-c,c.webp,Wine C\n"
                "slug-d,d.jpeg,Wine D\nslug-x,ignored.txt,Wine X\n",
                encoding="utf-8",
            )

            with (
                patch.object(main, "PHOTO_DIR", photo_dir),
                patch.object(main, "CATALOG_CSV", catalog_csv),
            ):
                main.image_groups.cache_clear()
                first = main.image_groups()
                main.image_groups.cache_clear()

                self.assertEqual(first, main.image_groups())
                self.assertEqual(sorted(map(len, first)), [1, 1, 2])
                self.assertTrue(all(path.suffix != ".txt" for group in first for path, _, _ in group))
                self.assertEqual(
                    {item["wine_name"] for folder in main.folders() for item in folder["files"]},
                    {"Wine A", "Wine B", "Wine C", "Wine D"},
                )

    def test_near_duplicates_get_weighted_priority_without_a_hard_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photo_dir = root / "photos"
            photo_dir.mkdir()
            catalog_rows = []
            for prefix in ("duplicate", "regular"):
                for index in range(12):
                    filename = f"image-{prefix}-{index:02}.jpg"
                    (photo_dir / filename).touch()
                    catalog_rows.append(f"{prefix}-{index:02},{filename},Wine {index}")

            catalog_csv = root / "catalog.csv"
            catalog_csv.write_text(
                "Slug,Название фото,Название вина\n" + "\n".join(catalog_rows),
                encoding="utf-8",
            )
            csv_path = root / "near_duplicates.csv"
            csv_path.write_text(
                "slug_1,slug_2\n"
                + "\n".join(f"duplicate-{index:02},duplicate-{index + 1:02}" for index in range(0, 12, 2)),
                encoding="utf-8",
            )

            with (
                patch.object(main, "PHOTO_DIR", photo_dir),
                patch.object(main, "CATALOG_CSV", catalog_csv),
                patch.object(main, "NEAR_DUPLICATES_CSV", csv_path),
            ):
                main.image_groups.cache_clear()
                groups = main.image_groups()

            early = tuple(item for group in groups for item in group[:4])
            self.assertGreater(sum(slug.startswith("duplicate") for _, slug, _ in early), len(early) / 2)
            self.assertTrue(any(slug.startswith("regular") for _, slug, _ in early))

    def test_safe_unique_save(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target_dir = Path(directory) / main.safe_name("slug/..")
            first = main.save_unique(target_dir, "photo.jpg", b"first")
            second = main.save_unique(target_dir, "photo.jpg", b"second")

            self.assertEqual(target_dir.name, "slug_")
            self.assertEqual(first.name, "photo.jpg")
            self.assertEqual(second.name, "photo_2.jpg")

    def test_labeled_status_comes_from_saved_folder_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            save_dir = Path(directory)
            (save_dir / main.safe_name("slug/a")).mkdir()
            groups = (((Path("a.jpg"), "slug/a", "Wine A"), (Path("b.jpg"), "slug-b", "Wine B")),)

            with (
                patch.object(main, "SAVE_DIR", save_dir),
                patch.object(main, "image_groups", return_value=groups),
            ):
                self.assertEqual(main.labeled_slugs(), {"slug/a"})
                self.assertEqual(main.labeled_files(), ["slug/a"])


if __name__ == "__main__":
    unittest.main()
