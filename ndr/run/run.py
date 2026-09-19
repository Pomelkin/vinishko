"""Run the NDR solution and preserve complete per-generation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import statistics
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from contracts import ContractError
from contracts import NOT_FOUND
from contracts import SCHEMA_VERSION
from contracts import prediction_slug


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_DATASET = HERE / "dataset"
DEFAULT_PREDICTOR = REPO / "ndr" / "solution" / "predictor.py"
SOLUTION_DIR = REPO / "ndr" / "solution"
DEFAULT_COMPARE_YEAR_MATTERS = SOLUTION_DIR / "prompts" / "compare_year_matters.txt"
DEFAULT_COMPARE_YEAR_NOT_MATTER = (
    SOLUTION_DIR / "prompts" / "compare_year_not_matter.txt"
)
DEFAULT_RESOLVE_MULTIPLE = SOLUTION_DIR / "prompts" / "resolve_multiple_same.txt"
DEFAULT_OUTPUT_MODELS = SOLUTION_DIR / "models.py"
DEFAULT_CONFIG = SOLUTION_DIR / "config.py"
DEFAULT_RESULTS = REPO / "ndr" / "results"
DEFAULT_ENV_FILE = REPO / ".env"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ndr.solution.config import SETTINGS  # noqa: E402


class RunnerError(RuntimeError):
    """Raised for runner configuration or dataset errors."""


ENV_KEY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def file_sha256(path: Path) -> str:
    """Return a SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL lines."""
    if not path.is_file():
        raise RunnerError(f"missing dataset file: {path}")
    records = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RunnerError(f"invalid JSON at {path}:{number}: {error}") from error
    return records


def load_predictor(path: Path) -> tuple[ModuleType, Callable[[dict[str, Any]], Any]]:
    """Load predict(request) from a Python source file."""
    if not path.is_file():
        raise RunnerError(f"predictor file does not exist: {path}")
    module_name = f"ndr_user_predictor_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RunnerError(f"cannot import predictor: {path}")
    module = importlib.util.module_from_spec(spec)
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    predict = getattr(module, "predict", None)
    if not callable(predict):
        raise RunnerError(f"{path} must define predict(request)")
    return module, predict


def absolute_repo_path(value: str) -> Path:
    """Resolve and constrain a generated repository-relative path."""
    path = (REPO / value).resolve()
    try:
        path.relative_to(REPO)
    except ValueError as error:
        raise RunnerError(f"dataset path escapes the repository: {value}") from error
    if not path.is_file():
        raise RunnerError(f"dataset path does not exist: {value}")
    return path


def percentile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile for a non-empty list."""
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def json_safe(value: Any) -> Any:
    """Preserve JSON values and represent unusual predictor values explicitly."""
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return {
            "_non_json_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "repr": repr(value),
        }
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically write an indented UTF-8 JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            f"{json.dumps(value, ensure_ascii=False, indent=2)}\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_dotenv(text: str, *, source: str = ".env") -> dict[str, str]:
    """Parse project dotenv assignments without an external dependency."""
    values: dict[str, str] = {}
    for number, original in enumerate(text.splitlines(), start=1):
        line = original.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export") and line[6:7].isspace():
            line = line[6:].lstrip()
        if "=" not in line:
            raise RunnerError(f"invalid dotenv assignment at {source}:{number}")
        key, raw = line.split("=", 1)
        key, raw = key.strip(), raw.strip()
        if not ENV_KEY_PATTERN.fullmatch(key):
            raise RunnerError(f"invalid dotenv key at {source}:{number}: {key!r}")
        if raw.startswith(("'", '"')):
            quote = raw[0]
            end = raw.find(quote, 1)
            if end < 0:
                raise RunnerError(f"unterminated dotenv quote at {source}:{number}")
            tail = raw[end + 1 :].strip()
            if tail and not tail.startswith("#"):
                raise RunnerError(f"unexpected dotenv text at {source}:{number}")
            value = raw[1:end]
        else:
            comment = next(
                (i for i, char in enumerate(raw) if char == "#" and (i == 0 or raw[i - 1].isspace())),
                None,
            )
            value = raw[:comment].rstrip() if comment is not None else raw
        values[key] = value
    return values


def load_env_file(path: Path) -> dict[str, Any]:
    """Load .env while preserving explicitly supplied process environment values."""
    resolved = path.resolve()
    metadata: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.is_file(),
        "loaded_keys": [],
    }
    if not metadata["exists"]:
        return metadata
    values = parse_dotenv(resolved.read_text(encoding="utf-8-sig"), source=str(resolved))
    loaded = []
    for key, value in values.items():
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    metadata["loaded_keys"] = sorted(loaded)
    return metadata


def prepare_results(results_dir: Path, force: bool) -> tuple[Path, Path, Path]:
    """Create an empty results target without leaving stale per-case documents."""
    run_path = results_dir / "run.json"
    metrics_path = results_dir / "metrics.json"
    by_case = results_dir / "by_case"
    existing_cases = sorted(by_case.glob("*.json")) if by_case.is_dir() else []
    if (run_path.exists() or metrics_path.exists() or existing_cases) and not force:
        raise RunnerError(
            f"results already exist under {results_dir}; use --force to replace them",
        )
    if force:
        run_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)
        for path in existing_cases:
            path.unlink()
    by_case.mkdir(parents=True, exist_ok=True)
    return run_path, metrics_path, by_case


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics useful while comparing prompts."""
    total = len(rows)
    answered_rows = [
        row for row in rows if row["prediction"]["slug"] != NOT_FOUND
    ]
    correct = sum(bool(row["prediction"]["correct"]) for row in rows)
    correct_answered = sum(
        bool(row["prediction"]["correct"]) for row in answered_rows
    )
    latencies = [float(row["prediction"]["latency_ms"]) for row in rows]
    return {
        "queries": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "answered": len(answered_rows),
        "not_found": total - len(answered_rows),
        "accuracy_when_answered": (
            correct_answered / len(answered_rows) if answered_rows else 0.0
        ),
        "contract_errors": sum(
            row["prediction"]["status"] == "contract_error" for row in rows
        ),
        "predictor_errors": sum(
            row["prediction"]["status"] == "predictor_error" for row in rows
        ),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.50) if latencies else 0.0,
            "p95": percentile(latencies, 0.95) if latencies else 0.0,
            "max": max(latencies, default=0.0),
        },
    }


def iter_model_calls(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Extract all model calls from the current solution trace."""
    response = row.get("predictor_response")
    trace = response.get("_trace") if isinstance(response, Mapping) else None
    if not isinstance(trace, Mapping):
        return []
    calls = [call for call in trace.get("comparison_calls", []) if isinstance(call, Mapping)]
    if isinstance(trace.get("resolution_call"), Mapping):
        calls.append(trace["resolution_call"])
    return calls


def add_numeric_usage(value: Any, totals: dict[str, int | float], prefix: str = "") -> None:
    """Recursively sum numeric usage leaves using dotted paths."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            add_numeric_usage(child, totals, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and prefix:
        totals[prefix] = totals.get(prefix, 0) + value


def quality_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return accuracy, coverage, and error counters."""
    report = build_report(rows)
    total = report["queries"]
    return {
        "queries": total,
        "correct": report["correct"],
        "accuracy": report["accuracy"],
        "answered": report["answered"],
        "not_found": report["not_found"],
        "coverage": report["answered"] / total if total else 0.0,
        "accuracy_when_answered": report["accuracy_when_answered"],
        "contract_errors": report["contract_errors"],
        "predictor_errors": report["predictor_errors"],
    }


def build_metrics(
    rows: list[dict[str, Any]], *, run_id: str, model: str | None, elapsed_seconds: float,
) -> dict[str, Any]:
    """Build final aggregate metrics for a completed run."""
    report = build_report(rows)
    calls = [call for row in rows for call in iter_model_calls(row)]
    usage: dict[str, int | float] = {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "completion_tokens_details.reasoning_tokens": 0, "cost": 0,
    }
    generations = 0
    for call in calls:
        trace = call.get("trace")
        if not isinstance(trace, Mapping):
            continue
        if isinstance(trace.get("generations"), list):
            generations += len(trace["generations"])
        raw = trace.get("raw_response")
        if isinstance(raw, Mapping):
            add_numeric_usage(raw.get("usage"), usage)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        input_record = row.get("input", {})
        order = input_record.get("candidate_order", []) if isinstance(input_record, Mapping) else []
        grouped.setdefault(len(order) if isinstance(order, list) else 0, []).append(row)
    elapsed = max(float(elapsed_seconds), 0.0)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at": utc_now(),
        "model": model,
        "quality": quality_metrics(rows),
        "latency_ms": report["latency_ms"],
        "elapsed_seconds": round(elapsed, 3),
        "throughput_queries_per_second": len(rows) / elapsed if elapsed else 0.0,
        "model_activity": {
            "requests": len(calls),
            "successful_requests": sum(call.get("status") == "ok" for call in calls),
            "failed_requests": sum(call.get("status") != "ok" for call in calls),
            "generations_returned": generations,
            "usage": dict(sorted(usage.items())),
        },
        "by_group_size": {
            str(size): {
                "quality": quality_metrics(group),
                "latency_ms": build_report(group)["latency_ms"],
            }
            for size, group in sorted(grouped.items())
        },
    }


def execute_case(
    *,
    run_id: str,
    case: dict[str, Any],
    request: dict[str, Any],
    input_record: dict[str, Any],
    predict: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    """Execute one predictor call while preserving every returned artifact."""
    started_at = utc_now()
    started = time.perf_counter()
    status = "ok"
    error_message = None
    raw_response: Any = None
    allowed = {candidate["slug"] for candidate in request["group"]["candidates"]}
    try:
        raw_response = predict(request)
        predicted = prediction_slug(raw_response, allowed)
        if isinstance(raw_response, Mapping):
            internal_status = raw_response.get("_status", "ok")
            if internal_status not in {"ok", "contract_error", "predictor_error"}:
                raise ContractError(f"unknown predictor status: {internal_status!r}")
            status = internal_status
            error_message = raw_response.get("_error")
            if status != "ok":
                predicted = NOT_FOUND
    except ContractError as error:
        predicted = NOT_FOUND
        status = "contract_error"
        error_message = str(error)
    except Exception as error:  # noqa: BLE001 - retain later cases and exception trace.
        predicted = NOT_FOUND
        status = "predictor_error"
        error_message = f"{type(error).__name__}: {error}"
        raw_response = {
            "exception": error_message,
            "trace": json_safe(getattr(error, "trace", None)),
        }

    latency_ms = (time.perf_counter() - started) * 1000
    return {
        "schema_version": 2,
        "run_id": run_id,
        "query_id": case["query_id"],
        "group_id": case["group_id"],
        "started_at": started_at,
        "finished_at": utc_now(),
        "input": input_record,
        "expected_slug": case["expected_slug"],
        "prediction": {
            "slug": predicted,
            "correct": predicted == case["expected_slug"],
            "status": status,
            "error": error_message,
            "latency_ms": round(latency_ms, 3),
        },
        "predictor_response": json_safe(raw_response),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor", type=Path, default=DEFAULT_PREDICTOR)
    parser.add_argument(
        "--compare-year-matters-prompt",
        type=Path,
        default=DEFAULT_COMPARE_YEAR_MATTERS,
    )
    parser.add_argument(
        "--compare-year-not-matter-prompt",
        type=Path,
        default=DEFAULT_COMPARE_YEAR_NOT_MATTER,
    )
    parser.add_argument(
        "--resolve-multiple-prompt",
        type=Path,
        default=DEFAULT_RESOLVE_MULTIPLE,
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--model")
    parser.add_argument("--api-base")
    parser.add_argument("--api-key-env", default=SETTINGS.openrouter.api_key_env)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        default=SETTINGS.generation.reasoning_effort,
    )
    parser.add_argument(
        "--max-completion-tokens",
        "--max-tokens",
        dest="max_completion_tokens",
        type=int,
        default=SETTINGS.generation.max_completion_tokens,
    )
    parser.add_argument("--temperature", type=float, default=SETTINGS.generation.temperature)
    parser.add_argument("--generations", type=int, default=SETTINGS.generation.generations)
    parser.add_argument("--timeout", type=float, default=SETTINGS.execution.timeout_seconds)
    parser.add_argument("--concurrency", type=int, default=SETTINGS.execution.concurrency)
    parser.add_argument("--seed", type=int, default=SETTINGS.execution.candidate_order_seed)
    parser.add_argument("--limit", type=int, default=SETTINGS.execution.limit)
    parser.add_argument("--force", action="store_true", default=SETTINGS.execution.force)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate numeric run controls before creating artifacts."""
    if args.limit is not None and args.limit < 1:
        raise RunnerError("--limit must be positive")
    if not 1 <= args.concurrency <= 64:
        raise RunnerError("--concurrency must be between 1 and 64")
    if not 1 <= args.generations <= 16:
        raise RunnerError("--generations must be between 1 and 16")
    if args.max_completion_tokens < 1:
        raise RunnerError("--max-completion-tokens must be positive")
    if not 0 <= args.temperature <= 2:
        raise RunnerError("--temperature must be between 0 and 2")
    if not 0 < args.timeout <= 3600:
        raise RunnerError("--timeout must be between 0 and 3600 seconds")


def main() -> int:
    """Execute selected cases concurrently and write lossless JSON artifacts."""
    args = parse_args()
    try:
        validate_args(args)
        environment = load_env_file(args.env_file)
        args.model = (
            args.model
            or os.environ.get("OPENROUTER_MODEL")
            or SETTINGS.openrouter.model
        )
        args.api_base = (
            args.api_base
            or os.environ.get("OPENROUTER_API_BASE")
            or SETTINGS.openrouter.api_base
        )
        manifest = read_jsonl(args.dataset / "manifest.jsonl")
        groups = {
            row["group_id"]: row for row in read_jsonl(args.dataset / "groups.jsonl")
        }
        catalog = {
            row["slug"]: row for row in read_jsonl(args.dataset / "catalog.jsonl")
        }
        dataset_metadata = json.loads(
            (args.dataset / "metadata.json").read_text(encoding="utf-8"),
        )
        module, predict = load_predictor(args.predictor.resolve())
        prompt_paths = {
            "compare_year_matters": args.compare_year_matters_prompt.resolve(),
            "compare_year_not_matter": args.compare_year_not_matter_prompt.resolve(),
            "resolve_multiple_same": args.resolve_multiple_prompt.resolve(),
        }
        prompts = {
            key: {"path": str(path), "content": path.read_text(encoding="utf-8")}
            for key, path in prompt_paths.items()
        }
        output_models_path = DEFAULT_OUTPUT_MODELS.resolve()
        if not output_models_path.is_file():
            raise RunnerError(f"missing Pydantic output models: {output_models_path}")
        config_path = DEFAULT_CONFIG.resolve()
        if not config_path.is_file():
            raise RunnerError(f"missing solution config: {config_path}")
        run_path, metrics_path, by_case_path = prepare_results(args.results_dir.resolve(), args.force)
    except (OSError, RunnerError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    selected = manifest[: args.limit] if args.limit else manifest
    run_id = f"ndr-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    generation = SETTINGS.generation.model_copy(
        update={
            "reasoning_effort": args.reasoning_effort,
            "max_completion_tokens": args.max_completion_tokens,
            "temperature": args.temperature,
            "generations": args.generations,
        },
    )
    runtime = {
        "model": args.model,
        "api_base": args.api_base,
        "api_key_env": args.api_key_env,
        "http_referer_env": SETTINGS.openrouter.http_referer_env,
        "app_title_env": SETTINGS.openrouter.app_title_env,
        "user_agent": SETTINGS.openrouter.user_agent,
        "generation": generation.request_payload(),
        "image_detail": generation.image_detail,
        "provider": SETTINGS.openrouter.routing.request_payload(),
        "timeout": args.timeout,
    }
    run_document: dict[str, Any] = {
        "schema_version": 2,
        "run": {
            "run_id": run_id,
            "status": "running",
            "started_at": utc_now(),
            "finished_at": None,
            "predictor_path": str(args.predictor.resolve()),
            "predictor_sha256": file_sha256(args.predictor.resolve()),
            "prompts": {
                key: {"path": str(path), "sha256": file_sha256(path)}
                for key, path in prompt_paths.items()
            },
            "output_models": {
                "path": str(output_models_path),
                "sha256": file_sha256(output_models_path),
            },
            "config": {
                "path": str(config_path),
                "sha256": file_sha256(config_path),
            },
            "dataset_path": str(args.dataset.resolve()),
            "dataset_metadata": dataset_metadata,
            "results_dir": str(args.results_dir.resolve()),
            "concurrency": args.concurrency,
            "seed": args.seed,
            "runtime": runtime,
            "environment": environment,
            "cases_total": len(selected),
            "cases_completed": 0,
        },
        "summary": build_report([]),
        "cases": [],
    }
    atomic_write_json(run_path, run_document)

    completed_by_index: dict[int, dict[str, Any]] = {}
    futures: dict[Future, int] = {}
    started = time.perf_counter()
    with ExitStack() as cleanup, ThreadPoolExecutor(
        max_workers=args.concurrency,
        thread_name_prefix="ndr-case",
    ) as executor:
        for index, case in enumerate(selected):
            source_image = absolute_repo_path(case["image_path"])
            anonymous_image = HERE / (
                f".tmp-ndr-query-{uuid.uuid4().hex}-{case['query_id']}"
                f"{source_image.suffix.lower()}"
            )
            shutil.copyfile(source_image, anonymous_image)
            cleanup.callback(anonymous_image.unlink, missing_ok=True)

            group = groups[case["group_id"]]
            slugs = list(group["slugs"])
            random.Random(f"{args.seed}:{case['query_id']}").shuffle(slugs)
            request_candidates = []
            recorded_candidates = []
            for slug in slugs:
                recorded_candidate = dict(catalog[slug])
                request_candidate = dict(recorded_candidate)
                request_candidate["reference_image_path"] = str(
                    absolute_repo_path(recorded_candidate["reference_image_path"]),
                )
                recorded_candidates.append(recorded_candidate)
                request_candidates.append(request_candidate)

            request = {
                "schema_version": SCHEMA_VERSION,
                "query": {
                    "query_id": case["query_id"],
                    "image_path": str(anonymous_image),
                },
                "group": {
                    "group_id": group["group_id"],
                    "candidates": request_candidates,
                },
                "prompts": prompts,
                "runtime": runtime,
            }
            input_record = {
                "query": {
                    "image_path": case["image_path"],
                    "image_sha256": case["image_sha256"],
                    "image_bytes": case["image_bytes"],
                },
                "group_id": group["group_id"],
                "candidate_order": slugs,
                "candidates": recorded_candidates,
                "prompts": prompts,
                "runtime": runtime,
            }
            future = executor.submit(
                execute_case,
                run_id=run_id,
                case=case,
                request=request,
                input_record=input_record,
                predict=predict,
            )
            futures[future] = index

        for future in as_completed(futures):
            index = futures[future]
            record = future.result()
            completed_by_index[index] = record
            atomic_write_json(by_case_path / f"{record['query_id']}.json", record)

            ordered = [completed_by_index[key] for key in sorted(completed_by_index)]
            run_document["cases"] = ordered
            run_document["summary"] = build_report(ordered)
            run_document["run"]["cases_completed"] = len(ordered)
            atomic_write_json(run_path, run_document)
            print(
                f"[{len(ordered)}/{len(selected)}] {record['query_id']}: "
                f"{record['prediction']['status']} -> {record['prediction']['slug']}",
                file=sys.stderr,
            )

    close = getattr(module, "close", None)
    if callable(close):
        close()

    ordered = [completed_by_index[key] for key in sorted(completed_by_index)]
    summary = build_report(ordered)
    has_errors = bool(summary["contract_errors"] or summary["predictor_errors"])
    elapsed_seconds = time.perf_counter() - started
    run_document["cases"] = ordered
    run_document["summary"] = summary
    run_document["run"].update(
        {
            "status": "completed_with_errors" if has_errors else "completed",
            "finished_at": utc_now(),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "cases_completed": len(ordered),
        },
    )
    atomic_write_json(run_path, run_document)
    atomic_write_json(
        metrics_path,
        build_metrics(ordered, run_id=run_id, model=args.model, elapsed_seconds=elapsed_seconds),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"run: {run_path}")
    print(f"metrics: {metrics_path}")
    print(f"by case: {by_case_path}")
    return 2 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
