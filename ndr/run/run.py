"""Run a named NDR solution and preserve versioned experiment artifacts."""

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
DEFAULT_SOLUTIONS = REPO / "ndr" / "solutions"
DEFAULT_RESULTS = REPO / "ndr" / "results"
DEFAULT_ENV_FILE = REPO / ".env"
REQUIRED_SOLUTION_FILES = (
    "config.py",
    "models.py",
    "predictor.py",
    "prompts/compare_year_matters.txt",
    "prompts/compare_year_not_matter.txt",
    "prompts/resolve_multiple_same.txt",
)
class RunnerError(RuntimeError):
    """Raised for runner configuration or dataset errors."""


ENV_KEY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
REASONING_EFFORT_NAMES = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"},
)
WINDOWS_TRANSIENT_REPLACE_ERRORS = frozenset({5, 32, 33})
ATOMIC_REPLACE_RETRY_SECONDS = 5.0
ATOMIC_REPLACE_RETRY_DELAY_SECONDS = 0.05


def parse_reasoning_effort(value: str) -> str | int:
    """Parse a named effort or DeepSeek's numeric 1-100 reasoning budget."""
    if value in REASONING_EFFORT_NAMES:
        return value
    try:
        effort = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "reasoning effort must be a named level or an integer from 1 to 100",
        ) from error
    if not 1 <= effort <= 100:
        raise argparse.ArgumentTypeError(
            "numeric reasoning effort must be between 1 and 100",
        )
    return effort


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


def load_solution_module(path: Path) -> ModuleType:
    """Load a module inside an isolated synthetic package.

    The package wrapper makes relative imports such as ``from .models`` work even
    when a solution folder was copied or its name contains punctuation.
    """
    if not path.is_file():
        raise RunnerError(f"Python module does not exist: {path}")
    package_digest = hashlib.sha256(str(path.parent.resolve()).encode()).hexdigest()[:16]
    package_name = f"_ndr_solution_{package_digest}"
    package = sys.modules.get(package_name)
    if package is None:
        package = ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = [str(path.parent.resolve())]
        sys.modules[package_name] = package
    module_name = f"{package_name}.{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RunnerError(f"cannot import Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_predictor(path: Path) -> tuple[ModuleType, Callable[[dict[str, Any]], Any]]:
    """Load predict(request) from a Python source file."""
    if not path.is_file():
        raise RunnerError(f"predictor file does not exist: {path}")
    module = load_solution_module(path)
    predict = getattr(module, "predict", None)
    if not callable(predict):
        raise RunnerError(f"{path} must define predict(request)")
    return module, predict


def solution_path(solutions_dir: Path, name: str) -> Path:
    """Resolve one direct-child solution folder name without path traversal."""
    if not name.strip() or name in {".", ".."} or Path(name).name != name:
        raise RunnerError("--solution must be one folder name, not a path")
    root = solutions_dir.resolve()
    path = (root / name).resolve()
    if path.parent != root or not path.is_dir():
        raise RunnerError(f"solution does not exist: {name!r} under {root}")
    missing = [relative for relative in REQUIRED_SOLUTION_FILES if not (path / relative).is_file()]
    if missing:
        raise RunnerError(
            f"solution {name!r} is incomplete; missing: {', '.join(missing)}",
        )
    return path


def discover_solutions(solutions_dir: Path) -> list[str]:
    """Return valid named solution folders in deterministic order."""
    if not solutions_dir.is_dir():
        return []
    names = []
    for child in solutions_dir.iterdir():
        if not child.is_dir():
            continue
        if all((child / relative).is_file() for relative in REQUIRED_SOLUTION_FILES):
            names.append(child.name)
    return sorted(names, key=str.casefold)


def load_solution_settings(path: Path) -> Any:
    """Load and minimally validate a solution's SETTINGS object."""
    module = load_solution_module(path / "config.py")
    settings = getattr(module, "SETTINGS", None)
    if settings is None:
        raise RunnerError(f"{path / 'config.py'} must define SETTINGS")
    for attribute in ("openrouter", "generation", "execution"):
        if not hasattr(settings, attribute):
            raise RunnerError(f"SETTINGS must define {attribute}")
    return settings


def solution_files(path: Path) -> list[Path]:
    """Enumerate source files included in a solution fingerprint."""
    return sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file()
            and "__pycache__" not in item.parts
            and item.suffix not in {".pyc", ".pyo"}
            and item.name != ".DS_Store"
        ),
        key=lambda item: item.relative_to(path).as_posix(),
    )


def solution_manifest(path: Path) -> tuple[str, dict[str, str]]:
    """Return a stable whole-solution fingerprint and per-file SHA-256 map."""
    digest = hashlib.sha256()
    files: dict[str, str] = {}
    for item in solution_files(path):
        relative = item.relative_to(path).as_posix()
        item_hash = file_sha256(item)
        files[relative] = item_hash
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), files


def run_timestamp(moment: datetime | None = None) -> str:
    """Return a sortable, Windows-safe UTC timestamp with collision headroom."""
    value = moment or datetime.now(UTC)
    return value.strftime("%Y%m%dT%H%M%S.%fZ")


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
        deadline = time.monotonic() + ATOMIC_REPLACE_RETRY_SECONDS
        while True:
            try:
                os.replace(temporary, path)
                break
            except PermissionError as error:
                # Windows denies replacement while Defender, an indexer, or an editor
                # briefly holds the destination without delete sharing. Concurrent runs
                # produce checkpoint bursts that make this otherwise transient lock likely.
                if (
                    getattr(error, "winerror", None)
                    not in WINDOWS_TRANSIENT_REPLACE_ERRORS
                    or time.monotonic() >= deadline
                ):
                    raise
                time.sleep(ATOMIC_REPLACE_RETRY_DELAY_SECONDS)
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


def prepare_results(
    results_root: Path,
    solution_name: str,
    timestamp: str,
) -> tuple[Path, Path, Path, Path]:
    """Create one never-overwritten ``solution/timestamp`` experiment directory."""
    solution_results = results_root.resolve() / solution_name
    solution_results.mkdir(parents=True, exist_ok=True)
    results_dir = solution_results / timestamp
    try:
        results_dir.mkdir()
    except FileExistsError as error:
        raise RunnerError(
            f"experiment directory already exists: {results_dir}; run again for a new timestamp",
        ) from error
    run_path = results_dir / "run.json"
    metrics_path = results_dir / "metrics.json"
    by_case = results_dir / "by_case"
    by_case.mkdir(parents=True, exist_ok=True)
    return results_dir, run_path, metrics_path, by_case


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics useful while comparing prompts."""
    attempted = len(rows)
    evaluated_rows = [
        row for row in rows if row["prediction"]["status"] == "ok"
    ]
    total = len(evaluated_rows)
    not_found = sum(
        row["prediction"]["slug"] == NOT_FOUND for row in evaluated_rows
    )
    correct = sum(bool(row["prediction"]["correct"]) for row in evaluated_rows)
    latencies = [float(row["prediction"]["latency_ms"]) for row in rows]
    return {
        "queries": total,
        "attempted_queries": attempted,
        "excluded_errors": attempted - total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "answered": total,
        "not_found": not_found,
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
    """Return verdict accuracy and error counters."""
    report = build_report(rows)
    return {
        "queries": report["queries"],
        "attempted_queries": report["attempted_queries"],
        "excluded_errors": report["excluded_errors"],
        "correct": report["correct"],
        "accuracy": report["accuracy"],
        "answered": report["answered"],
        "not_found": report["not_found"],
        "contract_errors": report["contract_errors"],
        "predictor_errors": report["predictor_errors"],
    }


def build_metrics(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    run_timestamp: str,
    solution_name: str,
    solution_fingerprint: str,
    model: str | None,
    elapsed_seconds: float,
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
        "run_timestamp": run_timestamp,
        "generated_at": utc_now(),
        "solution": {
            "name": solution_name,
            "fingerprint": solution_fingerprint,
        },
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
    predicted: str | None = None
    allowed = {candidate["slug"] for candidate in request["group"]["candidates"]}
    try:
        raw_response = predict(request)
        if isinstance(raw_response, Mapping):
            internal_status = raw_response.get("_status", "ok")
            if internal_status not in {"ok", "contract_error", "predictor_error"}:
                raise ContractError(f"unknown predictor status: {internal_status!r}")
            status = internal_status
            error_message = raw_response.get("_error")
        if status == "ok":
            predicted = prediction_slug(raw_response, allowed)
    except ContractError as error:
        predicted = None
        status = "contract_error"
        error_message = str(error)
    except Exception as error:  # noqa: BLE001 - retain later cases and exception trace.
        predicted = None
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
            "correct": predicted == case["expected_slug"] if status == "ok" else None,
            "status": status,
            "error": error_message,
            "latency_ms": round(latency_ms, 3),
        },
        "predictor_response": json_safe(raw_response),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solution",
        default="baseline",
        help="folder name under --solutions-dir (default: baseline)",
    )
    parser.add_argument("--solutions-dir", type=Path, default=DEFAULT_SOLUTIONS)
    parser.add_argument(
        "--list-solutions",
        action="store_true",
        help="list runnable solution folder names and exit",
    )
    parser.add_argument(
        "--predictor",
        type=Path,
        help="advanced one-run override; normally use the selected solution's predictor.py",
    )
    parser.add_argument(
        "--compare-year-matters-prompt",
        type=Path,
        help="advanced one-run override for the selected solution prompt",
    )
    parser.add_argument(
        "--compare-year-not-matter-prompt",
        type=Path,
        help="advanced one-run override for the selected solution prompt",
    )
    parser.add_argument(
        "--resolve-multiple-prompt",
        type=Path,
        help="advanced one-run override for the selected solution prompt",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help="experiment root; outputs go to <root>/<solution>/<UTC timestamp>",
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--model")
    parser.add_argument("--api-base")
    parser.add_argument("--api-key-env")
    parser.add_argument(
        "--reasoning-effort",
        type=parse_reasoning_effort,
        metavar="LEVEL|1..100",
    )
    parser.add_argument(
        "--max-completion-tokens",
        "--max-tokens",
        dest="max_completion_tokens",
        type=int,
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def apply_solution_defaults(args: argparse.Namespace, settings: Any) -> None:
    """Resolve config/env defaults while retaining explicit CLI precedence."""
    args.model = (
        args.model
        or os.environ.get("OPENROUTER_MODEL")
        or settings.openrouter.model
    )
    args.api_base = (
        args.api_base
        or os.environ.get("OPENROUTER_API_BASE")
        or settings.openrouter.api_base
    )
    if args.api_key_env is None:
        args.api_key_env = settings.openrouter.api_key_env
    generation_defaults = {
        "reasoning_effort": settings.generation.reasoning_effort,
        "max_completion_tokens": settings.generation.max_completion_tokens,
        "temperature": settings.generation.temperature,
        "generations": settings.generation.generations,
    }
    execution_defaults = {
        "timeout": settings.execution.timeout_seconds,
        "concurrency": settings.execution.concurrency,
        "seed": settings.execution.candidate_order_seed,
        "limit": settings.execution.limit,
    }
    for attribute, value in {**generation_defaults, **execution_defaults}.items():
        if getattr(args, attribute) is None:
            setattr(args, attribute, value)


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
    if args.list_solutions:
        for name in discover_solutions(args.solutions_dir.resolve()):
            print(name)
        return 0

    try:
        source_solution = solution_path(args.solutions_dir, args.solution)
        settings = load_solution_settings(source_solution)
        environment = load_env_file(args.env_file)
        apply_solution_defaults(args, settings)
        validate_args(args)
        dataset_path = args.dataset.resolve()
        manifest = read_jsonl(dataset_path / "manifest.jsonl")
        groups = {
            row["group_id"]: row for row in read_jsonl(dataset_path / "groups.jsonl")
        }
        catalog = {
            row["slug"]: row for row in read_jsonl(dataset_path / "catalog.jsonl")
        }
        dataset_metadata = json.loads(
            (dataset_path / "metadata.json").read_text(encoding="utf-8"),
        )

        timestamp = run_timestamp()
        results_dir, run_path, metrics_path, by_case_path = prepare_results(
            args.results_dir,
            args.solution,
            timestamp,
        )
        solution_fingerprint, solution_file_hashes = solution_manifest(source_solution)

        predictor_path = (
            args.predictor.resolve()
            if args.predictor is not None
            else source_solution / "predictor.py"
        )
        module, predict = load_predictor(predictor_path)
        prompt_paths = {
            "compare_year_matters": (
                args.compare_year_matters_prompt.resolve()
                if args.compare_year_matters_prompt is not None
                else source_solution / "prompts" / "compare_year_matters.txt"
            ),
            "compare_year_not_matter": (
                args.compare_year_not_matter_prompt.resolve()
                if args.compare_year_not_matter_prompt is not None
                else source_solution / "prompts" / "compare_year_not_matter.txt"
            ),
            "resolve_multiple_same": (
                args.resolve_multiple_prompt.resolve()
                if args.resolve_multiple_prompt is not None
                else source_solution / "prompts" / "resolve_multiple_same.txt"
            ),
        }
        prompts = {
            key: {"path": str(path), "content": path.read_text(encoding="utf-8")}
            for key, path in prompt_paths.items()
        }
        output_models_path = source_solution / "models.py"
        config_path = source_solution / "config.py"
    except (OSError, RunnerError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    selected = manifest[: args.limit] if args.limit else manifest
    run_id = f"ndr-{args.solution}-{timestamp}-{solution_fingerprint[:12]}"
    generation = settings.generation.model_copy(
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
        "http_referer_env": settings.openrouter.http_referer_env,
        "app_title_env": settings.openrouter.app_title_env,
        "user_agent": settings.openrouter.user_agent,
        "generation": generation.request_payload(),
        "image_detail": generation.image_detail,
        "provider": settings.openrouter.routing.request_payload(),
        "timeout": args.timeout,
    }
    run_document: dict[str, Any] = {
        "schema_version": 2,
        "run": {
            "run_id": run_id,
            "timestamp": timestamp,
            "status": "running",
            "started_at": utc_now(),
            "finished_at": None,
            "solution": {
                "name": args.solution,
                "source_path": str(source_solution),
                "fingerprint": solution_fingerprint,
                "files": solution_file_hashes,
            },
            "predictor_path": str(predictor_path),
            "predictor_sha256": file_sha256(predictor_path),
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
            "dataset_path": str(dataset_path),
            "dataset_metadata": dataset_metadata,
            "results_root": str(args.results_dir.resolve()),
            "results_dir": str(results_dir),
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
            prediction = record["prediction"]
            if prediction["status"] == "ok":
                outcome = f"ok -> {prediction['slug']}"
            else:
                outcome = f"{prediction['status']}: {prediction['error']}"
            print(
                f"[{len(ordered)}/{len(selected)}] {record['query_id']}: {outcome}",
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
        build_metrics(
            ordered,
            run_id=run_id,
            run_timestamp=timestamp,
            solution_name=args.solution,
            solution_fingerprint=solution_fingerprint,
            model=args.model,
            elapsed_seconds=elapsed_seconds,
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"run: {run_path}")
    print(f"metrics: {metrics_path}")
    print(f"by case: {by_case_path}")
    return 2 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
