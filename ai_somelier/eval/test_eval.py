# ruff: noqa: D101, D102
"""Offline tests for scaffold provenance and metric aggregation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ai_somelier.eval.common import DEFAULT_DATASET_PATH
from ai_somelier.eval.common import EvaluationError
from ai_somelier.eval.common import load_json
from ai_somelier.eval.metrics import compute_metrics
from ai_somelier.eval.scaffold import build_evaluation


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _fake_run(root: Path) -> Path:
    dataset = load_json(DEFAULT_DATASET_PATH)
    summaries = []
    for case_index, case in enumerate(dataset["cases"]):
        filename = f"cases/{case_index:04d}_{case['id']}.json"
        task_count = 1 + len(case["golden"]["turns"])
        turns = [
            {
                "turn_index": turn_index,
                "solution_response": {
                    "content": f"actual {case['id']} {turn_index}",
                    "suggestions": ["one", "two"] if turn_index == 0 else None,
                    "_status": "ok",
                    "_error": None,
                },
            }
            for turn_index in range(task_count)
        ]
        _write(
            root / filename,
            {
                "case_id": case["id"],
                "error": None,
                "turns": turns,
            },
        )
        summaries.append({"case_id": case["id"], "file": filename})
    _write(root / "run.json", {"run_id": "unit-eval", "cases": summaries})
    return root / "run.json"


def _complete(evaluation: dict) -> None:
    evaluation["status"] = "complete"
    for case in evaluation["cases"]:
        for task in case["tasks"]:
            for check in [*task["must_include"], *task["tone_of_voice"]]:
                check["passed"] = True
                check["evidence"] = "Подтверждено тестом."


class EvalTests(unittest.TestCase):
    def test_perfect_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evaluation = build_evaluation(_fake_run(Path(directory)))
            _complete(evaluation)
            report = compute_metrics(evaluation)

        self.assertEqual(report["summary"]["tasks"]["total"], 24)
        self.assertEqual(report["summary"]["must_include"]["total"], 251)
        self.assertEqual(report["summary"]["must_include"]["rate"], 1.0)
        self.assertEqual(report["summary"]["wrong_claims"]["count"], 0)
        self.assertEqual(report["summary"]["tone_of_voice"]["rate"], 1.0)
        self.assertEqual(report["summary"]["tasks"]["strict_pass_rate"], 1.0)

    def test_failure_and_wrong_claim_are_counted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evaluation = build_evaluation(_fake_run(Path(directory)))
            _complete(evaluation)
            task = evaluation["cases"][0]["tasks"][0]
            task["must_include"][0]["passed"] = False
            task["must_include"][0]["evidence"] = "Название отсутствует."
            task["tone_of_voice"][0]["passed"] = False
            task["tone_of_voice"][0]["evidence"] = "Нет прямого ответа."
            task["wrong_claims"] = [
                {
                    "id": "wc_01",
                    "claim": "Неверный факт.",
                    "evidence": "Цитата из ответа.",
                    "reason": "Противоречит карточке.",
                    "source": "input.wine",
                },
            ]
            report = compute_metrics(evaluation)

        self.assertEqual(report["summary"]["wrong_claims"]["count"], 1)
        self.assertEqual(report["summary"]["tasks"]["strict_passed"], 23)
        self.assertEqual(report["summary"]["tone_of_voice"]["passed"], 143)

    def test_draft_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evaluation = build_evaluation(_fake_run(Path(directory)))
            with self.assertRaises(EvaluationError):
                compute_metrics(evaluation)


if __name__ == "__main__":
    unittest.main()
