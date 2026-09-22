# ruff: noqa: D102
"""Tests for complete, ordered generation logs."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ai_somelier.run.run import DEFAULT_CONCURRENCY
from ai_somelier.run.run import run_experiment


def _dataset() -> dict[str, Any]:
    """Return a compact two-session golden dataset."""
    return {
        "schema_version": "test",
        "dataset_id": "runner-test",
        "cases": [
            {
                "id": "dialogue",
                "title": "Two follow-ups",
                "input": {
                    "wine": {"Slug": "recognized"},
                    "candidates": [{"Slug": "candidate"}],
                },
                "golden": {
                    "first_turn": {"content": "expected opening"},
                    "turns": [
                        {"user": "Первый вопрос", "assistant": "gold one"},
                        {"user": "Второй вопрос", "assistant": "gold two"},
                    ],
                },
            },
            {
                "id": "opening-only",
                "title": "No follow-ups",
                "input": {
                    "wine": {"Slug": "second"},
                    "candidates": [],
                },
                "golden": {
                    "first_turn": {"content": "another expected opening"},
                    "turns": [],
                },
            },
        ],
    }


def _responder(request: Mapping[str, Any]) -> dict[str, Any]:
    """Return raw messages with reasoning, like a successful solution."""
    history = request["history"]
    opening = not history
    message = {
        "role": "assistant",
        "content": "opening raw JSON" if opening else f"answer {len(history)}",
        "reasoning": "opening reasoning" if opening else "follow-up reasoning",
        "reasoning_details": [
            {
                "type": "reasoning.text",
                "text": "complete trace",
            },
        ],
    }
    return {
        "content": "display content",
        "suggestions": ["one", "two"] if opening else None,
        "message": message,
        "_status": "ok",
        "_error": None,
        "_trace": {
            "request": request,
            "raw_response": {"choices": [{"message": message}]},
        },
    }


class RunTests(unittest.TestCase):
    """Verify session ordering and persisted artifact completeness."""

    def test_default_concurrency_is_ten(self) -> None:
        self.assertEqual(DEFAULT_CONCURRENCY, 10)

    def test_run_keeps_full_inputs_reasoning_and_ordered_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir, run_record = run_experiment(
                dataset=_dataset(),
                dataset_path=root / "golden.json",
                dataset_sha256="dataset-hash",
                experiment_name="unit test",
                solution_name="fake",
                responder=_responder,
                results_dir=root / "results",
                concurrency=2,
                environment={"api_key_present": True},
            )

            self.assertTrue(output_dir.name.startswith("unit-test_"))
            self.assertEqual(run_record["status"], "completed")
            self.assertEqual(run_record["mode"], "generation_only")
            self.assertFalse(run_record["runner"]["scoring_enabled"])
            self.assertEqual(run_record["progress"]["successful_cases"], 2)

            saved_run = json.loads(
                (output_dir / "run.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(saved_run, run_record)
            dialogue_path = output_dir / saved_run["cases"][0]["file"]
            dialogue = json.loads(dialogue_path.read_text(encoding="utf-8"))

            self.assertEqual(dialogue["status"], "completed")
            self.assertEqual(len(dialogue["turns"]), 3)
            self.assertEqual(dialogue["turns_completed"], 3)
            self.assertEqual(dialogue["follow_up_turns_completed"], 2)
            self.assertEqual(
                dialogue["source_case"]["golden"]["turns"][0]["assistant"],
                "gold one",
            )
            self.assertNotIn("golden", dialogue["turns"][1]["solution_request"])
            self.assertEqual(
                dialogue["turns"][0]["solution_response"]["message"]["reasoning"],
                "opening reasoning",
            )
            self.assertEqual(
                dialogue["turns"][0]["solution_response"]["_trace"]["raw_response"]
                ["choices"][0]["message"]["reasoning_details"][0]["text"],
                "complete trace",
            )
            first_follow_up_history = dialogue["turns"][1]["solution_request"]["history"]
            self.assertEqual(
                [message["role"] for message in first_follow_up_history],
                ["assistant", "user"],
            )
            self.assertEqual(first_follow_up_history[0]["reasoning"], "opening reasoning")
            self.assertEqual(first_follow_up_history[1]["content"], "Первый вопрос")
            self.assertEqual(
                [message["role"] for message in dialogue["transcript"]],
                ["assistant", "user", "assistant", "user", "assistant"],
            )

    def test_failed_turn_stops_that_session_without_retry(self) -> None:
        calls = 0

        def fail(request: Mapping[str, Any]) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            return {
                "content": None,
                "suggestions": None,
                "message": None,
                "_status": "provider_error",
                "_error": "test failure",
                "_trace": {"request": request, "reasoning": "retained"},
            }

        dataset = _dataset()
        dataset["cases"] = dataset["cases"][:1]
        with tempfile.TemporaryDirectory() as directory:
            output_dir, run_record = run_experiment(
                dataset=dataset,
                dataset_path=Path(directory) / "golden.json",
                dataset_sha256="dataset-hash",
                experiment_name="failure",
                solution_name="fake",
                responder=fail,
                results_dir=Path(directory) / "results",
                concurrency=1,
            )
            case_path = output_dir / run_record["cases"][0]["file"]
            saved_case = json.loads(case_path.read_text(encoding="utf-8"))

        self.assertEqual(calls, 1)
        self.assertEqual(run_record["status"], "completed_with_errors")
        self.assertEqual(saved_case["status"], "failed")
        self.assertEqual(len(saved_case["turns"]), 1)
        self.assertEqual(
            saved_case["turns"][0]["solution_response"]["_trace"]["reasoning"],
            "retained",
        )


if __name__ == "__main__":
    unittest.main()
