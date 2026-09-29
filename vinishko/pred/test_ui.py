"""Похожие для бутылки, которую отверг поиск: вина категории whatis, своя винодельня — первой, выбор стабилен для бутылки."""

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
    **{f"abrau-white-{i}": row("Абрау-Дюрсо", "Белое") for i in range(3)},
    **{f"abrau-red-{i}": row("Абрау-Дюрсо", "Красное") for i in range(2)},
    **{f"other-white-{i}": row("Другая", "Белое") for i in range(6)},
    **{f"other-orange-{i}": row("Другая", "Оранжевое") for i in range(2)},
}


class SuggestionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = WineCatalog(
            ROWS, {slug: f"{slug}.jpg" for slug in ROWS if slug != "other-white-0"}
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

    def test_category_with_own_winery_first(self) -> None:
        title, slugs = self.slugs("Абрау-Дюрсо", "Белое")
        self.assertEqual(title, "Белое Абрау-Дюрсо в каталоге")
        self.assertEqual(len(slugs), SUGGESTIONS)
        self.assertEqual(sorted(slugs[:3]), [f"abrau-white-{i}" for i in range(3)])
        self.assertTrue(all(slug.startswith("other-white-") for slug in slugs[3:]))
        self.assertNotIn("other-white-0", slugs)

    def test_winery_without_that_category(self) -> None:
        title, slugs = self.slugs("Абрау-Дюрсо", "Оранжевое")
        self.assertEqual(title, "Оранжевые вина в каталоге")
        self.assertEqual(sorted(slugs), ["other-orange-0", "other-orange-1"])

    def test_unknown_winery_gives_category_only(self) -> None:
        title, slugs = self.slugs(UNKNOWN, "Белое")
        self.assertEqual(title, "Белые вина в каталоге")
        self.assertTrue(all("-white-" in slug for slug in slugs))

    def test_unknown_category_gives_winery(self) -> None:
        title, slugs = self.slugs("Абрау-Дюрсо", UNKNOWN)
        self.assertEqual(title, "Абрау-Дюрсо в каталоге")
        self.assertTrue(all(slug.startswith("abrau-") for slug in slugs))

    def test_nothing_recognized_gives_nothing(self) -> None:
        self.assertEqual(self.slugs(UNKNOWN, UNKNOWN), (None, []))
        self.assertEqual(self.slugs(None, None), (None, []))

    def test_choice_is_stable_per_bottle(self) -> None:
        self.assertEqual(
            self.slugs(UNKNOWN, "Белое", "a"), self.slugs(UNKNOWN, "Белое", "a")
        )


if __name__ == "__main__":
    unittest.main()
