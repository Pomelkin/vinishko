"""Regression checks for the v2 request and first-turn contract."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from ai_somelier.solution.v2.models import OutputContractError
from ai_somelier.solution.v2.models import validate_first_turn
from ai_somelier.solution.v2.sommelier import prepare_request
from ai_somelier.solution.v2.sommelier import respond


WINE = {
    "Название вина": "Проверочное вино",
    "Винодельня": "Проверочная винодельня",
    "Slug": "primary-wine",
}
CANDIDATE = {
    "Название вина": "Уникальный кандидат",
    "Slug": "unique-candidate",
}


class FakeHTTPResponse:
    status = 200

    def __init__(self, payload: dict[str, object]) -> None:
        self.body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def provider_response(content: str) -> FakeHTTPResponse:
    return FakeHTTPResponse({"choices": [{"message": {"content": content}}]})


class V2RequestTests(unittest.TestCase):
    def test_first_turn_omits_candidate_cards_and_counting_limits(self) -> None:
        prepared = prepare_request(
            {"wine": WINE, "candidates": [CANDIDATE], "history": []},
        )
        prompt = prepared.payload["messages"][0]["content"]
        schema = prepared.payload["response_format"]["json_schema"]["schema"]

        self.assertEqual(prepared.payload["max_completion_tokens"], 32768)
        self.assertIn("Проверочное вино", prompt)
        self.assertNotIn("Уникальный кандидат", prompt)
        self.assertNotIn("unique-candidate", prompt)
        self.assertNotIn("TOP-5 КАНДИДАТОВ", prompt)
        self.assertIn("4–5 коротких предложений", prompt)
        for forbidden in ("350–750", "100 слов", "30 знаков", "один знак вопроса"):
            self.assertNotIn(forbidden, prompt)
        self.assertNotIn("minLength", schema["properties"]["content"])
        self.assertNotIn("maxLength", schema["properties"]["content"])
        self.assertNotIn("maxLength", schema["properties"]["suggestions"]["items"])

    def test_every_follow_up_includes_candidate_cards(self) -> None:
        history = [
            {"role": "assistant", "content": "Предыдущий ответ."},
            {"role": "user", "content": "Что думаете?"},
        ]
        prepared = prepare_request(
            {"wine": WINE, "candidates": [CANDIDATE], "history": history},
        )
        prompt = prepared.payload["messages"][0]["content"]

        self.assertEqual(prepared.payload["max_completion_tokens"], 32768)
        self.assertIn("Уникальный кандидат", prompt)
        self.assertIn("unique-candidate", prompt)
        self.assertIn("TOP-5 КАНДИДАТОВ", prompt)
        self.assertNotIn("response_format", prepared.payload)

    def test_first_turn_validator_does_not_count_chars_or_words(self) -> None:
        content = "Длинное описание вина. " * 50 + "\nЧем я могу помочь?"
        suggestion = "Расскажите подробнее о подаче этого вина"
        output = validate_first_turn(
            {"content": content, "suggestions": [suggestion, "Что во вкусе?"]},
        )
        self.assertEqual(output["content"], content)
        self.assertEqual(output["suggestions"][0], suggestion)
        self.assertEqual(output["suggestions"][1], "Что во вкусе")

    def test_first_turn_requires_exact_closing_on_its_own_line(self) -> None:
        for content in (
            "Проверочное вино. Чем я могу помочь?",
            "Проверочное вино.\nЧем я могу вам помочь?",
        ):
            with self.subTest(content=content), self.assertRaises(OutputContractError):
                validate_first_turn(
                    {"content": content, "suggestions": ["Вкус", "Подача"]},
                )

    def test_suggestions_strip_trailing_question_marks_and_reject_empty(self) -> None:
        output = validate_first_turn(
            {
                "content": "Проверочное вино.\nЧем я могу помочь?",
                "suggestions": ["Что во вкусе?? ", "Как подавать？？"],
            },
        )
        self.assertEqual(output["suggestions"], ["Что во вкусе", "Как подавать"])
        with self.assertRaises(OutputContractError):
            validate_first_turn(
                {
                    "content": "Проверочное вино.\nЧем я могу помочь?",
                    "suggestions": ["?", "Как подавать"],
                },
            )


class V2RetryTests(unittest.TestCase):
    def test_one_retry_after_invalid_first_turn_json(self) -> None:
        valid = json.dumps(
            {
                "content": "Проверочное вино.\nЧем я могу помочь?",
                "suggestions": ["Что во вкусе?", "Как подавать?"],
            },
            ensure_ascii=False,
        )
        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
            patch(
                "ai_somelier.solution.v2.sommelier.urllib.request.urlopen",
                side_effect=[provider_response("not json"), provider_response(valid)],
            ) as send,
        ):
            result = respond({"wine": WINE, "candidates": [CANDIDATE], "history": []})

        self.assertEqual(send.call_count, 2)
        self.assertEqual(result["_status"], "ok")
        self.assertEqual(result["suggestions"], ["Что во вкусе", "Как подавать"])
        self.assertEqual(result["message"]["content"], valid)
        self.assertEqual(len(result["_trace"]["attempts"]), 2)
        self.assertIn("invalid first-turn JSON", result["_trace"]["attempts"][0]["validation_error"])
        self.assertEqual(
            result["_trace"]["attempts"][0]["raw_response"]["choices"][0]["message"]["content"],
            "not json",
        )

    def test_only_one_retry_after_schema_validation_failure(self) -> None:
        invalid = json.dumps(
            {"content": "Ответ без нужного окончания.", "suggestions": ["Вкус?", "Подача?"]},
            ensure_ascii=False,
        )
        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
            patch(
                "ai_somelier.solution.v2.sommelier.urllib.request.urlopen",
                side_effect=[provider_response(invalid), provider_response(invalid)],
            ) as send,
        ):
            result = respond({"wine": WINE, "candidates": [CANDIDATE], "history": []})

        self.assertEqual(send.call_count, 2)
        self.assertEqual(result["_status"], "contract_error")
        self.assertEqual(len(result["_trace"]["attempts"]), 2)
        self.assertTrue(all("validation_error" in attempt for attempt in result["_trace"]["attempts"]))

    def test_provider_error_is_not_retried(self) -> None:
        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
            patch(
                "ai_somelier.solution.v2.sommelier.urllib.request.urlopen",
                return_value=FakeHTTPResponse({"error": "provider refused request"}),
            ) as send,
        ):
            result = respond({"wine": WINE, "candidates": [CANDIDATE], "history": []})

        self.assertEqual(send.call_count, 1)
        self.assertEqual(result["_status"], "provider_error")
        self.assertEqual(len(result["_trace"]["attempts"]), 1)


if __name__ == "__main__":
    unittest.main()
