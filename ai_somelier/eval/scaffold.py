"""Create a complete draft evaluation aligned with one lossless run."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_somelier.eval.common import DEFAULT_CRITERIA_PATH
from ai_somelier.eval.common import DEFAULT_DATASET_PATH
from ai_somelier.eval.common import actual_cases
from ai_somelier.eval.common import display_path
from ai_somelier.eval.common import load_criteria
from ai_somelier.eval.common import load_dataset
from ai_somelier.eval.common import load_run
from ai_somelier.eval.common import run_fingerprint
from ai_somelier.eval.common import sha256
from ai_somelier.eval.common import write_json


def _must_include(case: Mapping[str, Any], turn_index: int) -> list[dict[str, str]]:
    golden = case["golden"]
    target = golden["first_turn"] if turn_index == 0 else golden["turns"][turn_index - 1]
    criteria = target.get("must_include")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError(f"{case['id']} turn {turn_index} has no must_include criteria")
    return criteria


def _checklist(items: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": item["id"],
            "criterion": item["criterion"],
            "passed": None,
            "evidence": "",
        }
        for item in items
    ]


def build_evaluation(
    run_path: Path,
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    criteria_path: Path = DEFAULT_CRITERIA_PATH,
) -> dict[str, Any]:
    """Build a draft with every expected binary decision represented once."""
    dataset_path = dataset_path.resolve()
    criteria_path = criteria_path.resolve()
    dataset = load_dataset(dataset_path)
    criteria = load_criteria(criteria_path)
    run_file, run, case_files = load_run(run_path)
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run.run_id must be a non-empty string")

    golden_by_id = {case["id"]: case for case in dataset["cases"]}
    cases = actual_cases(dataset, case_files)
    for case in cases:
        golden_case = golden_by_id[case["case_id"]]
        for task in case["tasks"]:
            task["must_include"] = _checklist(
                _must_include(golden_case, task["turn_index"]),
            )
            task["wrong_claims"] = []
            task["tone_of_voice"] = _checklist(criteria["tone_of_voice"])

    return {
        "schema_version": "1.0",
        "evaluation_id": f"{run_id}_codex_eval",
        "status": "draft",
        "created_at": datetime.now(UTC).isoformat(),
        "source": {
            "run_path": display_path(run_file),
            "run_id": run_id,
            "run_fingerprint_sha256": run_fingerprint(run_file, case_files),
            "dataset_path": display_path(dataset_path),
            "dataset_id": dataset.get("dataset_id"),
            "dataset_sha256": sha256(dataset_path),
            "tone_criteria_path": display_path(criteria_path),
            "tone_criteria_id": criteria["criteria_id"],
            "tone_criteria_sha256": sha256(criteria_path),
        },
        "cases": cases,
    }


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Add scaffold CLI arguments to a parser."""
    parser.add_argument("--run", type=Path, required=True, help="run directory or run.json")
    parser.add_argument("--output", type=Path, required=True, help="new draft evaluation JSON")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--criteria", type=Path, default=DEFAULT_CRITERIA_PATH)


def run_cli(args: argparse.Namespace) -> int:
    """Create the requested draft and print its path."""
    evaluation = build_evaluation(
        args.run,
        dataset_path=args.dataset,
        criteria_path=args.criteria,
    )
    write_json(args.output, evaluation)
    print(args.output.resolve())
    return 0


def main() -> int:
    """Run the standalone scaffold CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    configure_parser(parser)
    return run_cli(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
