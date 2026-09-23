"""Shared loading and provenance helpers for sommelier evaluation."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = ROOT / "ai_somelier" / "golden_dataset.json"
DEFAULT_CRITERIA_PATH = ROOT / "ai_somelier" / "eval" / "criteria.json"


class EvaluationError(ValueError):
    """Raised when an evaluation artifact violates its contract."""


def require(condition: bool, message: str) -> None:
    """Raise a stable validation error when a condition is false."""
    if not condition:
        raise EvaluationError(message)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object and reject duplicate keys."""
    resolved = path.resolve()
    require(resolved.is_file(), f"JSON file does not exist: {resolved}")
    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_unique_object,
        )
    except json.JSONDecodeError as error:
        raise EvaluationError(f"invalid JSON in {resolved}: {error}") from error
    require(isinstance(value, dict), f"JSON root must be an object: {resolved}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write a readable UTF-8 JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    """Return the SHA-256 of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_path(value: str | Path) -> Path:
    """Resolve an absolute path or a repository-relative path."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def display_path(path: Path) -> str:
    """Prefer a stable repository-relative POSIX path."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_dataset(path: Path) -> dict[str, Any]:
    """Load the golden dataset fields needed by evaluation."""
    dataset = load_json(path)
    cases = dataset.get("cases")
    require(isinstance(cases, list) and bool(cases), "dataset.cases must be a non-empty array")
    require(all(isinstance(case, dict) for case in cases), "every dataset case must be an object")
    return dataset


def load_criteria(path: Path) -> dict[str, Any]:
    """Load global tone-of-voice criteria."""
    criteria = load_json(path)
    require(
        set(criteria) == {"schema_version", "criteria_id", "tone_of_voice"},
        "tone criteria root has unexpected or missing keys",
    )
    items = criteria["tone_of_voice"]
    require(isinstance(items, list) and bool(items), "tone_of_voice must be a non-empty array")
    ids: set[str] = set()
    for index, item in enumerate(items):
        require(isinstance(item, dict), f"tone_of_voice[{index}] must be an object")
        require(
            set(item) == {"id", "criterion", "pass_when", "fail_when"},
            f"tone_of_voice[{index}] has unexpected or missing keys",
        )
        criterion_id = item["id"]
        require(isinstance(criterion_id, str) and bool(criterion_id), f"tone_of_voice[{index}].id is empty")
        require(criterion_id not in ids, f"duplicate tone criterion: {criterion_id}")
        ids.add(criterion_id)
        for field in ("criterion", "pass_when", "fail_when"):
            require(
                isinstance(item[field], str) and bool(item[field].strip()),
                f"tone_of_voice[{index}].{field} is empty",
            )
    return criteria


def resolve_run_file(path: Path) -> Path:
    """Resolve either a run directory or its run.json file."""
    resolved = path.resolve()
    run_file = resolved / "run.json" if resolved.is_dir() else resolved
    require(run_file.is_file(), f"run.json does not exist: {run_file}")
    return run_file


def _run_case_files(run_file: Path, run: Mapping[str, Any]) -> dict[str, Path]:
    summaries = run.get("cases")
    require(isinstance(summaries, list), "run.cases must be an array")
    result: dict[str, Path] = {}
    for index, summary in enumerate(summaries):
        require(isinstance(summary, Mapping), f"run.cases[{index}] must be an object")
        case_id = summary.get("case_id")
        relative = summary.get("file")
        require(isinstance(case_id, str) and bool(case_id), f"run.cases[{index}].case_id is empty")
        require(isinstance(relative, str) and bool(relative), f"run.cases[{index}].file is empty")
        require(case_id not in result, f"duplicate run case: {case_id}")
        case_file = (run_file.parent / relative).resolve()
        require(case_file.is_file(), f"run case file does not exist: {case_file}")
        result[case_id] = case_file
    return result


def load_run(path: Path) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    """Load a run manifest and resolve every referenced case artifact."""
    run_file = resolve_run_file(path)
    run = load_json(run_file)
    return run_file, run, _run_case_files(run_file, run)


def run_fingerprint(run_file: Path, case_files: Mapping[str, Path]) -> str:
    """Fingerprint run.json and every case file in stable case-ID order."""
    digest = hashlib.sha256()
    files = [("run.json", run_file), *sorted(case_files.items())]
    for label, path in files:
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _actual_response(case_result: Mapping[str, Any], turn_index: int) -> dict[str, Any]:
    turns = case_result.get("turns")
    if not isinstance(turns, list):
        turns = []
    turn = next(
        (
            item
            for item in turns
            if isinstance(item, Mapping) and item.get("turn_index") == turn_index
        ),
        None,
    )
    response = turn.get("solution_response") if isinstance(turn, Mapping) else None
    if not isinstance(response, Mapping):
        response = {}
    content = response.get("content")
    suggestions = response.get("suggestions")
    if not isinstance(content, str):
        content = None
    if not isinstance(suggestions, list) or not all(isinstance(item, str) for item in suggestions):
        suggestions = None
    status = response.get("_status")
    if not isinstance(status, str):
        status = "missing"
    error = response.get("_error")
    if not isinstance(error, str):
        error = case_result.get("error")
    if not isinstance(error, str):
        error = None
    return {
        "content": content,
        "suggestions": suggestions,
        "generation_status": status,
        "generation_error": error,
    }


def actual_cases(
    dataset: Mapping[str, Any],
    case_files: Mapping[str, Path],
) -> list[dict[str, Any]]:
    """Return actual responses aligned to every golden task, including missing turns."""
    result = []
    for case_index, case in enumerate(dataset["cases"]):
        case_id = case.get("id")
        require(isinstance(case_id, str) and bool(case_id), f"dataset case {case_index} has no ID")
        require(case_id in case_files, f"run is missing case: {case_id}")
        case_result = load_json(case_files[case_id])
        require(case_result.get("case_id") == case_id, f"case artifact ID differs: {case_id}")
        golden = case.get("golden")
        require(isinstance(golden, Mapping), f"{case_id}.golden must be an object")
        turns = golden.get("turns")
        require(isinstance(turns, list), f"{case_id}.golden.turns must be an array")
        tasks = [
            {
                "turn_index": 0,
                "kind": "opening",
                "user_message": None,
                "actual_response": _actual_response(case_result, 0),
            },
        ]
        for turn_index, turn in enumerate(turns, start=1):
            require(isinstance(turn, Mapping), f"{case_id}.golden.turns[{turn_index - 1}] must be an object")
            tasks.append(
                {
                    "turn_index": turn_index,
                    "kind": "follow_up",
                    "user_message": turn.get("user"),
                    "actual_response": _actual_response(case_result, turn_index),
                },
            )
        result.append({"case_id": case_id, "tasks": tasks})
    return result
