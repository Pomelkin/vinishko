"""Подборка вин каталога для бутылки не из каталога: цепочка признаков, стабильный выбор, позиции с фото первыми."""

import unittest

from vinishko.pred.ui import SUGGESTIONS
from vinishko.pred.ui import WineCatalog
from vinishko.whatis.solution.catalog import UNKNOWN


def row(winery: str, category: str) -> dict[str, str]:
    return {
        "Название вина": f"{winery} {category}",
        "Винодельня": winery,
        "Категория": category,
    }


ROWS = {
    **{f"abrau-red-{i}": row("Абрау-Дюрсо", "Красное") for i in range(7)},
    **{f"abrau-white-{i}": row("Абрау-Дюрсо", "Белое") for i in range(3)},
    **{f"other-orange-{i}": row("Другая", "Оранжевое") for i in range(2)},
}


class SuggestionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = WineCatalog(
            ROWS, {slug: f"{slug}.jpg" for slug in ROWS if slug != "abrau-red-0"}
        )

    def slugs(
        self, brand: str | None, category: str | None, seed: str = "bottle"
    ) -> tuple[str | None, list[str]]:
        found = self.catalog.suggestions(brand, category, seed)
        return (
            (found["title"], [w["slug"] for w in found["wines"]])
            if found
            else (None, [])
        )

    def test_brand_and_color_first(self) -> None:
        title, slugs = self.slugs("Абрау-Дюрсо", "Красное")
        self.assertEqual(title, "Красное Абрау-Дюрсо в каталоге")
        self.assertEqual(len(slugs), SUGGESTIONS)
        self.assertTrue(all(slug.startswith("abrau-red-") for slug in slugs))
        self.assertNotIn("abrau-red-0", slugs)

    def test_brand_without_that_color_falls_back_to_brand(self) -> None:
        title, slugs = self.slugs("Абрау-Дюрсо", "Оранжевое")
        self.assertEqual(title, "Абрау-Дюрсо в каталоге")
        self.assertTrue(all(slug.startswith("abrau-") for slug in slugs))

    def test_unknown_brand_falls_back_to_color(self) -> None:
        title, slugs = self.slugs(UNKNOWN, "Оранжевое")
        self.assertEqual(title, "Оранжевые вина в каталоге")
        self.assertEqual(sorted(slugs), ["other-orange-0", "other-orange-1"])

    def test_nothing_recognized_gives_nothing(self) -> None:
        self.assertEqual(self.slugs(UNKNOWN, UNKNOWN), (None, []))
        self.assertEqual(self.slugs(None, None), (None, []))

    def test_choice_is_stable_per_bottle(self) -> None:
        self.assertEqual(
            self.slugs("Абрау-Дюрсо", None, "a"), self.slugs("Абрау-Дюрсо", None, "a")
        )


if __name__ == "__main__":
    unittest.main()
