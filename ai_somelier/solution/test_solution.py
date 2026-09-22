# ruff: noqa: D102
"""Unit tests for prompt construction and expert selection."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_somelier.solution.config import DEFAULT_KNOWLEDGE_PATH
from ai_somelier.solution.knowledge import NO_DATA
from ai_somelier.solution.knowledge import PROMPT_CARD_FIELDS
from ai_somelier.solution.knowledge import TEMPLATE_PATH
from ai_somelier.solution.knowledge import compile_expert_knowledge
from ai_somelier.solution.knowledge import load_expert_knowledge
from ai_somelier.solution.knowledge import normalize_catalog_card
from ai_somelier.solution.knowledge import selected_knowledge_values
from ai_somelier.solution.sommelier import prepare_request


def wine_card(**overrides: str | None) -> dict[str, str | None]:
    """Return a compact Catalog-shaped fixture."""
    card: dict[str, str | None] = {
        "Название вина": "Тестовое вино",
        "Категория": "Белое",
        "Цвет": "",
        "Регион": "Крым",
        "Сорт винограда": "Рислинг",
        "Описание": "Цитрусовое, свежее, с заметной кислотностью.",
        "Винодельня": "Массандра",
        "Slug": "test-wine",
        "Винтаж": None,
        "Крепость, %": "12,5",
        "Крепость от, %": "",
        "Крепость до, %": "",
        "Сахар": "сухое",
        "Игристое": "не определено",
        "Метод игристого": "",
        "Креплёное": "",
        "Безалкогольное": "",
        "Органическое": "",
        "Выдержка или резерв": "",
    }
    card.update(overrides)
    return card


class KnowledgeTests(unittest.TestCase):
    """Verify normalization and recognized-wine-only retrieval."""

    def test_missing_values_are_visible(self) -> None:
        normalized = normalize_catalog_card(wine_card())
        self.assertEqual(normalized["Цвет"], NO_DATA)
        self.assertEqual(normalized["Винтаж"], NO_DATA)
        self.assertEqual(normalized["Игристое"], "не определено")

    def test_every_missing_representation_becomes_no_data(self) -> None:
        missing_values = (None, "", "   ", float("nan"))
        card = {
            field: missing_values[index % len(missing_values)]
            for index, field in enumerate(PROMPT_CARD_FIELDS)
        }
        self.assertEqual(
            set(normalize_catalog_card(card).values()),
            {NO_DATA},
        )

    def test_axes_preserve_unknown_positive_flags(self) -> None:
        selected = selected_knowledge_values(wine_card())
        self.assertEqual(selected["effervescence"], "unknown")
        self.assertEqual(selected["fortified"], "unknown")
        self.assertEqual(selected["organic"], "unknown")
        self.assertEqual(selected["alcohol_band"], "standard")

    def test_only_recognized_wine_blocks_are_compiled(self) -> None:
        template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
        template["color"]["white"] = ["Знание о белом вине."]
        template["color"]["red"] = ["Не должно попасть в промпт."]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge.json"
            path.write_text(
                json.dumps(template, ensure_ascii=False),
                encoding="utf-8",
            )
            knowledge = load_expert_knowledge(path)
        _, blocks = compile_expert_knowledge(wine_card(), knowledge)
        self.assertEqual(
            blocks,
            [
                {
                    "axis": "color",
                    "value": "white",
                    "bullets": ["Знание о белом вине."],
                },
            ],
        )

    def test_default_knowledge_is_filled_and_valid(self) -> None:
        knowledge = load_expert_knowledge(DEFAULT_KNOWLEDGE_PATH)
        self.assertTrue(
            all(bullets for values in knowledge.values() for bullets in values.values()),
        )

    def test_fully_missing_wine_uses_only_safe_unknown_blocks(self) -> None:
        missing_card: dict[str, None] = dict.fromkeys(PROMPT_CARD_FIELDS)
        knowledge = load_expert_knowledge(DEFAULT_KNOWLEDGE_PATH)
        selections, blocks = compile_expert_knowledge(missing_card, knowledge)
        self.assertEqual(set(selections.values()), {"unknown"})
        self.assertEqual(len(blocks), 10)
        self.assertNotIn("color", {block["axis"] for block in blocks})
        self.assertNotIn("region", {block["axis"] for block in blocks})
        self.assertNotIn("brand", {block["axis"] for block in blocks})
        self.assertTrue(all(block["value"] == "unknown" for block in blocks))


class PromptTests(unittest.TestCase):
    """Verify dynamic prompt and output-contract switching."""

    def test_first_turn_has_schema_at_the_absolute_end(self) -> None:
        prepared = prepare_request(
            {"wine": wine_card(), "candidates": [wine_card()], "history": []},
        )
        system_prompt = prepared.payload["messages"][0]["content"]
        schema = prepared.payload["response_format"]["json_schema"]["schema"]
        schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
        self.assertTrue(prepared.first_turn)
        self.assertTrue(system_prompt.endswith(schema_text))
        self.assertEqual(
            schema["properties"]["suggestions"]["maxItems"],
            2,
        )
        self.assertIn(f'"Винтаж": "{NO_DATA}"', system_prompt)

    def test_follow_up_is_plain_text_and_preserves_reasoning(self) -> None:
        reasoning_details = [{"type": "reasoning.text", "text": "trace"}]
        history = [
            {
                "role": "assistant",
                "content": '{"content":"prior","suggestions":["a","b"]}',
                "reasoning": "reasoning text",
                "reasoning_details": reasoning_details,
            },
            {"role": "user", "content": "Подойдёт к утке?"},
        ]
        prepared = prepare_request(
            {"wine": wine_card(), "candidates": [], "history": history},
        )
        self.assertFalse(prepared.first_turn)
        self.assertNotIn("response_format", prepared.payload)
        self.assertEqual(
            prepared.payload["messages"][1]["reasoning"],
            "reasoning text",
        )
        self.assertIs(
            prepared.payload["messages"][1]["reasoning_details"],
            reasoning_details,
        )

    def test_missing_cards_are_grounded_without_candidate_knowledge(self) -> None:
        missing_card: dict[str, None] = dict.fromkeys(PROMPT_CARD_FIELDS)
        candidate = wine_card(
            **{
                "Категория": "Красное",
                "Регион": "Кубань",
                "Сорт винограда": "Саперави",
                "Винодельня": "Фанагория",
            },
        )
        prepared = prepare_request(
            {"wine": missing_card, "candidates": [candidate], "history": []},
        )
        prompt = prepared.payload["messages"][0]["content"]
        self.assertGreaterEqual(prompt.count(NO_DATA), len(PROMPT_CARD_FIELDS))
        self.assertEqual(len(prepared.injected_expert_blocks), 10)
        self.assertTrue(
            all(
                block["value"] == "unknown"
                for block in prepared.injected_expert_blocks
            ),
        )


if __name__ == "__main__":
    unittest.main()
