"""Run versioned AI sommelier solutions over the golden dialogue dataset."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import json
import os
import platform
import re
import sys
import time
import traceback
from collections.abc import Callable
from collections.abc import Mapping
from datetime import UTC
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from dotenv import load_dotenv


AI_SOMELIER_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = AI_SOMELIER_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_DATASET_PATH = AI_SOMELIER_DIR / "golden_dataset.json"
DEFAULT_RESULTS_DIR = AI_SOMELIER_DIR / "results"
DEFAULT_ENV_PATH = REPO_ROOT / ".env"
DEFAULT_CONCURRENCY = 10
SOLUTION_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
UNSAFE_EXPERIMENT_CHARS = re.compile(r"[^\w.-]+", flags=re.UNICODE)
Respond = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class RunnerError(RuntimeError):
    """Raised for runner configuration or dataset contract failures."""


def _utc_now() -> str:
    """Return an unambiguous JSON timestamp."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _directory_timestamp() -> str:
    """Return a Windows-safe, collision-resistant UTC timestamp."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _sha256(path: Path) -> str:
    """Hash one file without loading it fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    """Atomically write one readable UTF-8 JSON artifact."""
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_experiment_name(value: str) -> str:
    """Convert a human experiment name to one safe path component."""
    normalized = UNSAFE_EXPERIMENT_CHARS.sub("-", value.strip()).strip(". -")
    if not normalized:
        raise RunnerError("experiment name must contain a letter or number")
    return normalized


def _positive_int(value: str) -> int:
    """Parse a positive CLI integer."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _load_dataset(path: Path) -> tuple[dict[str, Any], str]:
    """Load and minimally validate the golden dataset."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise RunnerError(f"dataset does not exist: {resolved}")
    try:
        dataset = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise RunnerError(f"invalid dataset JSON: {error}") from error
    if not isinstance(dataset, dict):
        raise RunnerError("dataset root must be an object")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RunnerError("dataset.cases must be a non-empty array")
    if any(not isinstance(case, dict) for case in cases):
        raise RunnerError("every dataset case must be an object")
    return dataset, _sha256(resolved)


def _load_solution(name: str) -> tuple[ModuleType, Respond]:
    """Import one versioned solution and validate its runner entry point."""
    if not SOLUTION_NAME.fullmatch(name):
        raise RunnerError(
            "solution must be a Python package name such as 'v1'",
        )
    module_name = f"ai_somelier.solution.{name}"
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as error:
        raise RunnerError(f"cannot import solution {module_name}: {error}") from error
    responder = getattr(module, "respond", None)
    if not callable(responder):
        raise RunnerError(f"solution {module_name} must export callable respond(request)")
    return module, responder


def _file_manifest(directory: Path) -> list[dict[str, Any]]:
    """Describe source files needed to identify an exact runner or solution."""
    manifest = []
    if not directory.is_dir():
        return manifest
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        manifest.append(
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            },
        )
    return manifest


def _questions(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read ordered user follow-ups without exposing expected answers to the model."""
    golden = case.get("golden")
    if not isinstance(golden, Mapping):
        raise RunnerError("case.golden must be an object")
    turns = golden.get("turns")
    if not isinstance(turns, list):
        raise RunnerError("case.golden.turns must be an array")
    normalized = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, Mapping):
            raise RunnerError(f"case.golden.turns[{index}] must be an object")
        user = turn.get("user")
        if not isinstance(user, str) or not user.strip():
            raise RunnerError(
                f"case.golden.turns[{index}].user must be a non-empty string",
            )
        normalized.append({"source_turn_index": index, "user": user})
    return normalized


def _call_turn(
    responder: Respond,
    request: dict[str, Any],
    *,
    turn_index: int,
    kind: str,
    user_message: str | None,
    source_turn_index: int | None,
) -> dict[str, Any]:
    """Call one model turn and retain its complete result or Python exception."""
    started_at = _utc_now()
    started = time.perf_counter()
    response: Mapping[str, Any] | None = None
    exception: dict[str, str] | None = None
    try:
        response = responder(request)
        if not isinstance(response, Mapping):
            raise TypeError("solution respond() must return an object")
        response = dict(response)
    except Exception as error:
        exception = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    return {
        "turn_index": turn_index,
        "kind": kind,
        "source_turn_index": source_turn_index,
        "user_message": user_message,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "solution_request": request,
        "solution_response": response,
        "exception": exception,
    }


def _turn_message(turn: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Return the raw assistant message or a reason that the session must stop."""
    exception = turn.get("exception")
    if isinstance(exception, Mapping):
        return None, f"{exception.get('type')}: {exception.get('message')}"
    response = turn.get("solution_response")
    if not isinstance(response, Mapping):
        return None, "solution returned no response object"
    status = response.get("_status")
    if status != "ok":
        return None, f"solution status {status!r}: {response.get('_error')}"
    message = response.get("message")
    if not isinstance(message, Mapping):
        return None, "successful solution response has no raw assistant message"
    return dict(message), None


def _run_case(
    index: int,
    case: dict[str, Any],
    *,
    run_id: str,
    solution_name: str,
    responder: Respond,
) -> dict[str, Any]:
    """Generate one stateful dialogue, keeping turns ordered inside the case."""
    started_at = _utc_now()
    started = time.perf_counter()
    case_id = case.get("id")
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "solution": solution_name,
        "case_index": index,
        "case_id": case_id,
        "title": case.get("title"),
        "status": "running",
        "error": None,
        "started_at": started_at,
        "finished_at": None,
        "duration_ms": None,
        "source_case": case,
        "session_input": None,
        "turns": [],
        "transcript": [],
        "turns_completed": 0,
        "follow_up_turns_completed": 0,
    }
    history: list[dict[str, Any]] = []
    try:
        if not isinstance(case_id, str) or not case_id.strip():
            raise RunnerError("case.id must be a non-empty string")
        input_data = case.get("input")
        if not isinstance(input_data, Mapping):
            raise RunnerError("case.input must be an object")
        wine = input_data.get("wine")
        candidates = input_data.get("candidates")
        if not isinstance(wine, Mapping):
            raise RunnerError("case.input.wine must be an object")
        if not isinstance(candidates, list):
            raise RunnerError("case.input.candidates must be an array")
        questions = _questions(case)
        result["session_input"] = {
            "wine": dict(wine),
            "candidates": candidates,
            "ordered_user_questions": [item["user"] for item in questions],
        }

        opening_request = {
            "wine": dict(wine),
            "candidates": candidates,
            "history": [],
        }
        opening = _call_turn(
            responder,
            opening_request,
            turn_index=0,
            kind="opening",
            user_message=None,
            source_turn_index=None,
        )
        result["turns"].append(opening)
        message, error = _turn_message(opening)
        if error is not None or message is None:
            result["status"] = "failed"
            result["error"] = error
            return result
        history.append(message)
        result["turns_completed"] += 1

        for turn_index, question in enumerate(questions, start=1):
            user_message = {
                "role": "user",
                "content": question["user"],
            }
            history.append(user_message)
            request = {
                "wine": dict(wine),
                "candidates": candidates,
                "history": list(history),
            }
            turn = _call_turn(
                responder,
                request,
                turn_index=turn_index,
                kind="follow_up",
                user_message=question["user"],
                source_turn_index=question["source_turn_index"],
            )
            result["turns"].append(turn)
            message, error = _turn_message(turn)
            if error is not None or message is None:
                result["status"] = "failed"
                result["error"] = error
                return result
            history.append(message)
            result["turns_completed"] += 1
            result["follow_up_turns_completed"] += 1
        result["status"] = "completed"
        return result
    except Exception as error:
        result["status"] = "runner_error"
        result["error"] = f"{type(error).__name__}: {error}"
        result["runner_exception"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        return result
    finally:
        result["transcript"] = history
        result["finished_at"] = _utc_now()
        result["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)


def _case_filename(index: int, case_id: Any) -> str:
    """Build a stable JSON filename independent of case execution order."""
    safe_id = UNSAFE_EXPERIMENT_CHARS.sub("-", str(case_id)).strip(". -") or "case"
    return f"{index:04d}_{safe_id}.json"


def _case_summary(case_result: Mapping[str, Any], relative_path: str) -> dict[str, Any]:
    """Return the non-evaluative per-case row stored in run.json."""
    return {
        "case_index": case_result.get("case_index"),
        "case_id": case_result.get("case_id"),
        "status": case_result.get("status"),
        "error": case_result.get("error"),
        "turns_completed": case_result.get("turns_completed"),
        "follow_up_turns_completed": case_result.get("follow_up_turns_completed"),
        "duration_ms": case_result.get("duration_ms"),
        "file": relative_path,
    }


def run_experiment(
    *,
    dataset: dict[str, Any],
    dataset_path: Path,
    dataset_sha256: str,
    experiment_name: str,
    solution_name: str,
    responder: Respond,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    concurrency: int = DEFAULT_CONCURRENCY,
    environment: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run all cases concurrently and write complete, generation-only JSON logs."""
    if concurrency < 1:
        raise RunnerError("concurrency must be at least 1")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RunnerError("dataset.cases must be a non-empty array")

    safe_experiment_name = _safe_experiment_name(experiment_name)
    run_id = f"{safe_experiment_name}_{_directory_timestamp()}"
    resolved_results_dir = results_dir.resolve()
    output_dir = resolved_results_dir / run_id
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=False)

    dataset_metadata = {key: value for key, value in dataset.items() if key != "cases"}
    solution_dir = AI_SOMELIER_DIR / "solution" / solution_name
    started_at = _utc_now()
    started = time.perf_counter()
    run_record: dict[str, Any] = {
        "schema_version": "1.0",
        "mode": "generation_only",
        "run_id": run_id,
        "experiment_name": experiment_name,
        "directory_experiment_name": safe_experiment_name,
        "output_directory": str(output_dir),
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "duration_seconds": None,
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": dataset_sha256,
            "metadata": dataset_metadata,
            "case_count": len(cases),
            "case_ids": [case.get("id") if isinstance(case, Mapping) else None for case in cases],
        },
        "solution": {
            "name": solution_name,
            "module": f"ai_somelier.solution.{solution_name}",
            "files": _file_manifest(solution_dir),
        },
        "runner": {
            "concurrency": concurrency,
            "retry_count": 0,
            "scoring_enabled": False,
            "files": _file_manifest(Path(__file__).resolve().parent),
            "python": sys.version,
            "platform": platform.platform(),
            "process_id": os.getpid(),
            "working_directory": str(Path.cwd()),
            "command": sys.argv,
        },
        "environment": dict(environment or {}),
        "progress": {
            "total_cases": len(cases),
            "finished_cases": 0,
            "successful_cases": 0,
            "failed_cases": 0,
        },
        "cases": [],
    }
    run_path = output_dir / "run.json"
    _write_json(run_path, run_record)

    summaries: dict[int, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        pending = {
            executor.submit(
                _run_case,
                index,
                case,
                run_id=run_id,
                solution_name=solution_name,
                responder=responder,
            ): (index, case)
            for index, case in enumerate(cases)
        }
        for future in concurrent.futures.as_completed(pending):
            index, case = pending[future]
            try:
                case_result = future.result()
            except Exception as error:
                case_result = {
                    "schema_version": "1.0",
                    "run_id": run_id,
                    "solution": solution_name,
                    "case_index": index,
                    "case_id": case.get("id"),
                    "status": "runner_error",
                    "error": f"{type(error).__name__}: {error}",
                    "source_case": case,
                    "runner_exception": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                }
            filename = _case_filename(index, case.get("id"))
            relative_path = f"cases/{filename}"
            _write_json(cases_dir / filename, case_result)
            summaries[index] = _case_summary(case_result, relative_path)
            ordered_summaries = [summaries[key] for key in sorted(summaries)]
            successful = sum(item["status"] == "completed" for item in ordered_summaries)
            run_record["cases"] = ordered_summaries
            run_record["progress"] = {
                "total_cases": len(cases),
                "finished_cases": len(ordered_summaries),
                "successful_cases": successful,
                "failed_cases": len(ordered_summaries) - successful,
            }
            _write_json(run_path, run_record)

    failed = run_record["progress"]["failed_cases"]
    run_record["status"] = "completed" if failed == 0 else "completed_with_errors"
    run_record["finished_at"] = _utc_now()
    run_record["duration_seconds"] = round(time.perf_counter() - started, 3)
    _write_json(run_path, run_record)
    return output_dir, run_record


def _parser() -> argparse.ArgumentParser:
    """Build the generation-only CLI."""
    parser = argparse.ArgumentParser(
        description="Generate complete AI sommelier dialogue logs without scoring.",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="experiment name used as the result-directory prefix",
    )
    parser.add_argument(
        "--solution",
        default="v1",
        help="package under ai_somelier/solution (default: v1)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="golden dataset JSON path",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="parent directory for timestamped result folders",
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=DEFAULT_CONCURRENCY,
        help=f"independent cases to run concurrently (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help="dotenv file loaded without overriding process variables",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return nonzero when any case fails to generate."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        dataset, dataset_sha256 = _load_dataset(args.dataset)
        _, responder = _load_solution(args.solution)
        env_path = args.env_file.resolve()
        env_loaded = load_dotenv(dotenv_path=env_path, override=False) if env_path.is_file() else False
        output_dir, run_record = run_experiment(
            dataset=dataset,
            dataset_path=args.dataset,
            dataset_sha256=dataset_sha256,
            experiment_name=args.experiment,
            solution_name=args.solution,
            responder=responder,
            results_dir=args.results_dir,
            concurrency=args.concurrency,
            environment={
                "dotenv_path": str(env_path),
                "dotenv_exists": env_path.is_file(),
                "dotenv_loaded": env_loaded,
                "api_key_present": bool(os.environ.get("OPENROUTER_API_KEY")),
            },
        )
    except (OSError, RunnerError) as error:
        parser.error(str(error))
    print(output_dir)
    return 0 if run_record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
