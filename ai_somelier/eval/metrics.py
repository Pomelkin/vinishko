"""Validate a completed structured evaluation and calculate JSON metrics."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import UTC
from datetime import datetime
from pathlib import Path
import re
from typing import Any

from ai_somelier.eval.common import EvaluationError
from ai_somelier.eval.common import actual_cases
from ai_somelier.eval.common import load_criteria
from ai_somelier.eval.common import load_dataset
from ai_somelier.eval.common import load_json
from ai_somelier.eval.common import load_run
from ai_somelier.eval.common import require
from ai_somelier.eval.common import resolve_path
from ai_somelier.eval.common import run_fingerprint
from ai_somelier.eval.common import sha256
from ai_somelier.eval.common import write_json


ROOT_KEYS = {"schema_version", "evaluation_id", "status", "created_at", "source", "cases"}
SOURCE_KEYS = {
    "run_path",
    "run_id",
    "run_fingerprint_sha256",
    "dataset_path",
    "dataset_id",
    "dataset_sha256",
    "tone_criteria_path",
    "tone_criteria_id",
    "tone_criteria_sha256",
}
TASK_KEYS = {
    "turn_index",
    "kind",
    "user_message",
    "actual_response",
    "must_include",
    "wrong_claims",
    "tone_of_voice",
}
CHECK_KEYS = {"id", "criterion", "passed", "evidence"}
WRONG_CLAIM_KEYS = {"id", "claim", "evidence", "reason", "source"}
WRONG_CLAIM_ID = re.compile(r"wc_[0-9]{2,}")


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _text(value: Any, location: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{location} must be non-empty text")
    return value


def _source_context(evaluation: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    source = evaluation.get("source")
    require(isinstance(source, dict) and set(source) == SOURCE_KEYS, "source has unexpected or missing keys")
    run_file, run, case_files = load_run(resolve_path(_text(source["run_path"], "source.run_path")))
    dataset_path = resolve_path(_text(source["dataset_path"], "source.dataset_path"))
    criteria_path = resolve_path(_text(source["tone_criteria_path"], "source.tone_criteria_path"))
    dataset = load_dataset(dataset_path)
    criteria = load_criteria(criteria_path)

    require(source["run_id"] == run.get("run_id"), "source.run_id differs from run.json")
    require(source["run_fingerprint_sha256"] == run_fingerprint(run_file, case_files), "run fingerprint differs")
    require(source["dataset_id"] == dataset.get("dataset_id"), "source.dataset_id differs")
    require(source["dataset_sha256"] == sha256(dataset_path), "golden dataset SHA-256 differs")
    require(source["tone_criteria_id"] == criteria["criteria_id"], "tone criteria ID differs")
    require(source["tone_criteria_sha256"] == sha256(criteria_path), "tone criteria SHA-256 differs")
    return dataset, criteria, actual_cases(dataset, case_files)


def _golden_criteria(case: Mapping[str, Any], turn_index: int) -> list[dict[str, str]]:
    golden = case["golden"]
    target = golden["first_turn"] if turn_index == 0 else golden["turns"][turn_index - 1]
    result = target.get("must_include")
    require(isinstance(result, list) and bool(result), f"{case['id']} turn {turn_index} has no must_include")
    return result


def _validate_checks(
    actual: Any,
    expected: list[Mapping[str, Any]],
    location: str,
) -> tuple[int, int, bool]:
    require(isinstance(actual, list), f"{location} must be an array")
    require(len(actual) == len(expected), f"{location} criterion count differs")
    passed = 0
    for index, (item, target) in enumerate(zip(actual, expected, strict=True)):
        item_location = f"{location}[{index}]"
        require(isinstance(item, dict) and set(item) == CHECK_KEYS, f"{item_location} has unexpected or missing keys")
        require(item["id"] == target["id"], f"{item_location}.id differs")
        require(item["criterion"] == target["criterion"], f"{item_location}.criterion differs")
        require(type(item["passed"]) is bool, f"{item_location}.passed must be boolean")
        _text(item["evidence"], f"{item_location}.evidence")
        passed += int(item["passed"])
    total = len(expected)
    return passed, total, passed == total


def _validate_wrong_claims(value: Any, location: str) -> int:
    require(isinstance(value, list), f"{location} must be an array")
    ids: set[str] = set()
    claims: set[str] = set()
    for index, item in enumerate(value):
        item_location = f"{location}[{index}]"
        require(isinstance(item, dict) and set(item) == WRONG_CLAIM_KEYS, f"{item_location} has unexpected or missing keys")
        claim_id = _text(item["id"], f"{item_location}.id")
        require(WRONG_CLAIM_ID.fullmatch(claim_id) is not None, f"{item_location}.id must match wc_NN")
        require(claim_id not in ids, f"{location} contains duplicate ID {claim_id}")
        ids.add(claim_id)
        claim = _text(item["claim"], f"{item_location}.claim")
        require(claim not in claims, f"{location} contains duplicate claim {claim!r}")
        claims.add(claim)
        for field in ("evidence", "reason", "source"):
            _text(item[field], f"{item_location}.{field}")
    return len(value)


def _task_metrics(
    task: Mapping[str, Any],
    golden_case: Mapping[str, Any],
    tone_criteria: list[Mapping[str, Any]],
    location: str,
) -> dict[str, Any]:
    must_passed, must_total, must_all = _validate_checks(
        task["must_include"],
        _golden_criteria(golden_case, task["turn_index"]),
        f"{location}.must_include",
    )
    tone_passed, tone_total, tone_all = _validate_checks(
        task["tone_of_voice"],
        tone_criteria,
        f"{location}.tone_of_voice",
    )
    wrong_count = _validate_wrong_claims(task["wrong_claims"], f"{location}.wrong_claims")
    if task["actual_response"]["content"] is None:
        require(must_passed == 0, f"{location}: missing answer cannot pass must_include")
        require(tone_passed == 0, f"{location}: missing answer cannot pass tone_of_voice")
        require(wrong_count == 0, f"{location}: missing answer cannot contain wrong claims")
    wrong_free = wrong_count == 0
    return {
        "turn_index": task["turn_index"],
        "kind": task["kind"],
        "must_include_passed": must_passed,
        "must_include_total": must_total,
        "must_include_all_passed": must_all,
        "wrong_claims_count": wrong_count,
        "wrong_claim_free": wrong_free,
        "tone_passed": tone_passed,
        "tone_total": tone_total,
        "tone_all_passed": tone_all,
        "strict_pass": must_all and wrong_free and tone_all,
    }


def _validate_task_identity(task: Any, expected: Mapping[str, Any], location: str) -> None:
    require(isinstance(task, dict) and set(task) == TASK_KEYS, f"{location} has unexpected or missing keys")
    for field in ("turn_index", "kind", "user_message", "actual_response"):
        require(task[field] == expected[field], f"{location}.{field} differs from source run")


def _summaries(case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = [task for case in case_reports for task in case["tasks"]]
    task_total = len(tasks)
    must_passed = sum(task["must_include_passed"] for task in tasks)
    must_total = sum(task["must_include_total"] for task in tasks)
    must_tasks = sum(task["must_include_all_passed"] for task in tasks)
    wrong_count = sum(task["wrong_claims_count"] for task in tasks)
    wrong_free = sum(task["wrong_claim_free"] for task in tasks)
    tone_passed = sum(task["tone_passed"] for task in tasks)
    tone_total = sum(task["tone_total"] for task in tasks)
    tone_tasks = sum(task["tone_all_passed"] for task in tasks)
    strict_tasks = sum(task["strict_pass"] for task in tasks)
    return {
        "tasks": {
            "total": task_total,
            "strict_passed": strict_tasks,
            "strict_pass_rate": _rate(strict_tasks, task_total),
        },
        "must_include": {
            "passed": must_passed,
            "total": must_total,
            "rate": _rate(must_passed, must_total),
            "tasks_all_passed": must_tasks,
            "task_pass_rate": _rate(must_tasks, task_total),
        },
        "wrong_claims": {
            "count": wrong_count,
            "tasks_without_wrong_claims": wrong_free,
            "wrong_claim_free_task_rate": _rate(wrong_free, task_total),
        },
        "tone_of_voice": {
            "passed": tone_passed,
            "total": tone_total,
            "rate": _rate(tone_passed, tone_total),
            "tasks_all_passed": tone_tasks,
            "task_pass_rate": _rate(tone_tasks, task_total),
        },
    }


def compute_metrics(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a completed evaluation and return aggregate and per-task metrics."""
    require(set(evaluation) == ROOT_KEYS, "evaluation root has unexpected or missing keys")
    require(evaluation["schema_version"] == "1.0", "unsupported evaluation schema version")
    _text(evaluation["evaluation_id"], "evaluation_id")
    _text(evaluation["created_at"], "created_at")
    require(evaluation["status"] == "complete", "evaluation status must be complete")
    dataset, criteria, expected_cases = _source_context(evaluation)
    cases = evaluation["cases"]
    require(isinstance(cases, list) and len(cases) == len(expected_cases), "evaluation case count differs")
    golden_by_id = {case["id"]: case for case in dataset["cases"]}
    case_reports = []
    for case_index, (case, expected) in enumerate(zip(cases, expected_cases, strict=True)):
        location = f"cases[{case_index}]"
        require(isinstance(case, dict) and set(case) == {"case_id", "tasks"}, f"{location} has unexpected or missing keys")
        require(case["case_id"] == expected["case_id"], f"{location}.case_id differs")
        tasks = case["tasks"]
        require(isinstance(tasks, list) and len(tasks) == len(expected["tasks"]), f"{location}.tasks count differs")
        task_reports = []
        for task_index, (task, expected_task) in enumerate(zip(tasks, expected["tasks"], strict=True)):
            task_location = f"{location}.tasks[{task_index}]"
            _validate_task_identity(task, expected_task, task_location)
            task_reports.append(
                _task_metrics(
                    task,
                    golden_by_id[case["case_id"]],
                    criteria["tone_of_voice"],
                    task_location,
                ),
            )
        case_reports.append(
            {
                "case_id": case["case_id"],
                "strict_tasks_passed": sum(item["strict_pass"] for item in task_reports),
                "tasks_total": len(task_reports),
                "tasks": task_reports,
            },
        )
    return {
        "schema_version": "1.0",
        "evaluation_id": evaluation["evaluation_id"],
        "source_run_id": evaluation["source"]["run_id"],
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": _summaries(case_reports),
        "cases": case_reports,
    }


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Add metrics CLI arguments to a parser."""
    parser.add_argument("evaluation", type=Path, help="completed evaluation JSON")
    parser.add_argument("--output", type=Path, help="write metrics JSON instead of stdout only")


def run_cli(args: argparse.Namespace) -> int:
    """Validate, calculate, and emit metrics."""
    report = compute_metrics(load_json(args.evaluation))
    if args.output is not None:
        write_json(args.output, report)
        print(args.output.resolve())
    else:
        import json

        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    """Run the standalone metrics CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    configure_parser(parser)
    try:
        return run_cli(parser.parse_args())
    except (EvaluationError, OSError, ValueError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
