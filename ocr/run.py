"""Run, parse, and evaluate baidu/Unlimited-OCR without mutating raw output."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = Path(__file__).resolve().parent / "dataset.jsonl"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_MODEL = "baidu/Unlimited-OCR"
UNLIMITED_OCR_PROMPT = "<image>document parsing."
PREPROCESS_MODES = ("original", "center-label", "label-scan")
ROLE_WEIGHTS = {"identity": 3.0, "disambiguation": 3.0, "secondary": 1.0}
MIME_BY_FORMAT = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}

REF_BLOCK_RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>", re.DOTALL)
DET_BLOCK_RE = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL)
SPECIAL_END_TOKENS = (
    "<｜end▁of▁sentence｜>",
    "<|end▁of▁sentence|>",
    "<|end_of_sentence|>",
)


class RunError(RuntimeError):
    """Raised for invalid runner configuration or an unusable server response."""


def load_environment(path: Path = ROOT / ".env") -> bool:
    """Load the repository ``.env`` without overriding process variables."""
    return load_dotenv(dotenv_path=path, override=False)


def parse_args() -> argparse.Namespace:
    """Parse the unified run/parse/evaluate CLI options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--raw-input",
        type=Path,
        help="Parse and evaluate an existing raw JSONL without calling the model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Raw model output (default: .../unlimited-ocr-TIMESTAMP.raw.jsonl)",
    )
    parser.add_argument("--parsed-output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--base-url", default=os.getenv("OCR_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("OCR_API_KEY") or "EMPTY")
    parser.add_argument("--model", default=os.getenv("OCR_MODEL") or DEFAULT_MODEL)
    parser.add_argument(
        "--timeout", type=float, default=float(os.getenv("OCR_TIMEOUT") or "120")
    )
    parser.add_argument(
        "--max-tokens", type=int, default=int(os.getenv("OCR_MAX_TOKENS") or "8192")
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--query-id", action="append", default=[])
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("OCR_CONCURRENCY") or "1"),
        help="Number of simultaneous API requests (default: 1)",
    )
    parser.add_argument(
        "--preprocess",
        choices=PREPROCESS_MODES,
        default=os.getenv("OCR_PREPROCESS") or "original",
        help=(
            "Image view sent to OCR: original, one center-label crop, or "
            "original plus three label crops (default: original)"
        ),
    )
    parser.add_argument(
        "--only-predicted",
        action="store_true",
        help="Score only present query IDs (for partial/smoke runs)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace parsed/report outputs; raw JSONL is never replaced",
    )
    return parser.parse_args()


def load_dataset(
    path: Path, query_ids: set[str], limit: int | None
) -> list[dict[str, Any]]:
    """Load selected rows without exposing gold features to the OCR model."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RunError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if query_ids and record["query_id"] not in query_ids:
                continue
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise RunError("No dataset records selected")
    unknown_ids = query_ids - {record["query_id"] for record in records}
    if unknown_ids:
        raise RunError(f"Unknown or excluded query IDs: {sorted(unknown_ids)}")
    return records


def image_data_url(path: Path) -> str:
    """Build a data URL using the image signature, not its filename extension."""
    with Image.open(path) as image:
        image_format = image.format
    if image_format is None:
        raise RunError(f"Pillow could not detect the image format: {path}")
    try:
        mime_type = MIME_BY_FORMAT[image_format]
    except KeyError as exc:
        raise RunError(f"Unsupported image format {image_format!r}: {path}") from exc
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _jpeg_data_url(image: Image.Image) -> str:
    """Encode a generated crop as a high-quality, widely supported JPEG URL."""
    output = io.BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=95, subsampling=0)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _fractional_crop(
    image: Image.Image, box: tuple[float, float, float, float]
) -> Image.Image:
    """Crop an image using normalized coordinates with non-empty pixel bounds."""
    left, top, right, bottom = box
    pixel_box = (
        max(0, min(image.width - 1, round(left * image.width))),
        max(0, min(image.height - 1, round(top * image.height))),
        max(1, min(image.width, round(right * image.width))),
        max(1, min(image.height, round(bottom * image.height))),
    )
    if pixel_box[2] <= pixel_box[0] or pixel_box[3] <= pixel_box[1]:
        raise RunError(f"Invalid crop box {box} for image size {image.size}")
    return image.crop(pixel_box)


def image_variants(path: Path, preprocess: str) -> list[tuple[str, str]]:
    """Build single-image OCR views without triggering vLLM multi-image mode.

    Unlimited-OCR is a document parser. On product photos it often emits one
    ``image`` region and deliberately omits text inside that region. Tight label
    views test whether that layout decision, rather than character recognition,
    is the limiting factor.
    """
    if preprocess == "original":
        return [("original", image_data_url(path))]

    crop_boxes = [
        ("center-label", (0.03, 0.12, 0.97, 0.88)),
    ]
    if preprocess == "label-scan":
        crop_boxes = [
            ("upper-label", (0.02, 0.00, 0.98, 0.70)),
            ("center-label", (0.02, 0.15, 0.98, 0.85)),
            ("lower-label", (0.02, 0.30, 0.98, 1.00)),
        ]

    with Image.open(path) as source:
        source.load()
        variants = [
            (name, _jpeg_data_url(_fractional_crop(source, box)))
            for name, box in crop_boxes
        ]
    if preprocess == "label-scan":
        return [("original", image_data_url(path)), *variants]
    return variants


def default_output_path() -> Path:
    """Create a Windows-safe timestamped raw filename in ``ocr/results``."""
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_RESULTS_DIR / f"unlimited-ocr-{timestamp}.raw.jsonl"


def derived_output_path(raw_path: Path, kind: str) -> Path:
    """Derive a sibling parsed/report path without reusing the raw path."""
    name = raw_path.name
    if name.endswith(".raw.jsonl"):
        base = name[: -len(".raw.jsonl")]
    elif raw_path.suffix:
        base = name[: -len(raw_path.suffix)]
    else:
        base = name
    suffix = ".parsed.jsonl" if kind == "parsed" else ".report.json"
    return raw_path.with_name(base + suffix)


def clean_unlimited_ocr_output(raw_output: str) -> str:
    """Remove Unlimited-OCR grounding markup while preserving transcribed text.

    The vLLM recipe emits text in ``ref`` blocks followed by coordinate-only ``det``
    blocks. The model-card postprocessor also documents lines where a ``det`` marker
    precedes ordinary content. Unwrapping refs and dropping only paired det blocks
    handles both forms without discarding the text after a det marker.
    """
    text = REF_BLOCK_RE.sub(lambda match: match.group(1), raw_output)
    text = DET_BLOCK_RE.sub("\n", text)
    for token in SPECIAL_END_TOKENS:
        text = text.replace(token, "")

    cleaned_lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    compacted: list[str] = []
    for line in cleaned_lines:
        if line or not compacted or compacted[-1]:
            compacted.append(line)
    return "\n".join(compacted).strip()


def build_request(model: str, data_url: str, max_tokens: int) -> dict[str, Any]:
    """Build the exact single-image request required by the official vLLM recipe."""
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": UNLIMITED_OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "extra_body": {
            "skip_special_tokens": False,
            "vllm_xargs": {"ngram_size": 35, "window_size": 128},
        },
    }


def usage_dict(usage: Any) -> dict[str, int] | None:
    """Convert SDK token usage to plain JSON when the server supplies it."""
    if usage is None:
        return None
    return {
        "prompt_tokens": int(usage.prompt_tokens or 0),
        "completion_tokens": int(usage.completion_tokens or 0),
        "total_tokens": int(usage.total_tokens or 0),
    }


def _sum_usage(attempts: list[dict[str, Any]]) -> dict[str, int] | None:
    """Sum token usage across planned image views when vLLM reports it."""
    usages = [attempt["usage"] for attempt in attempts if attempt["usage"] is not None]
    if not usages:
        return None
    return {
        key: sum(usage[key] for usage in usages)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _merge_text(attempts: list[dict[str, Any]]) -> str:
    """Join distinct non-empty parsed transcriptions while preserving view order."""
    blocks: list[str] = []
    seen: set[str] = set()
    for attempt in attempts:
        block = attempt["raw_text"].strip()
        if block and block not in seen:
            blocks.append(block)
            seen.add(block)
    return "\n\n".join(blocks)


def recognize(
    client: OpenAI,
    model: str,
    record: dict[str, Any],
    max_tokens: int,
    preprocess: str = "original",
) -> dict[str, Any]:
    """Run all planned single-image views and preserve unparsed responses."""
    started = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    image_path = ROOT / record["image_path"]
    try:
        variants = image_variants(image_path, preprocess)
    except Exception as exc:
        variants = []
        setup_error = f"{type(exc).__name__}: {exc}"
    else:
        setup_error = None

    response_model = model
    for variant, data_url in variants:
        attempt_started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                **build_request(model, data_url, max_tokens)
            )
            response_model = response.model or response_model
            choice = response.choices[0]
            raw_output = choice.message.content or ""
            if not raw_output.strip():
                raise RunError(
                    "Empty model output; verify the literal <image> prompt, "
                    "skip_special_tokens=false, and the Unlimited-OCR vLLM image"
                )
            attempts.append(
                {
                    "variant": variant,
                    "raw_output": raw_output,
                    "latency_ms": round(
                        (time.perf_counter() - attempt_started) * 1000, 3
                    ),
                    "finish_reason": choice.finish_reason,
                    "usage": usage_dict(response.usage),
                    "error": None,
                }
            )
        except Exception as exc:  # each planned view is attempted exactly once
            attempts.append(
                {
                    "variant": variant,
                    "raw_output": "",
                    "latency_ms": round(
                        (time.perf_counter() - attempt_started) * 1000, 3
                    ),
                    "finish_reason": None,
                    "usage": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    errors = list(
        dict.fromkeys(
            attempt["error"] for attempt in attempts if attempt["error"] is not None
        )
    )
    if setup_error is not None:
        errors.insert(0, setup_error)
    finish_reasons = [
        attempt["finish_reason"]
        for attempt in attempts
        if attempt["finish_reason"] is not None
    ]
    finish_reason = None
    if "length" in finish_reasons:
        finish_reason = "length"
    elif finish_reasons:
        finish_reason = finish_reasons[-1]

    return {
        "query_id": record["query_id"],
        "image_path": record["image_path"],
        "model": response_model,
        "preprocess": preprocess,
        "raw_output": "\n\n".join(
            attempt["raw_output"] for attempt in attempts if attempt["raw_output"]
        ),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "finish_reason": finish_reason,
        "usage": _sum_usage(attempts),
        "error": (
            None
            if any(attempt["raw_output"] for attempt in attempts)
            else "; ".join(errors) or "no model output"
        ),
        "attempts": attempts,
    }


def percentile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile without third-party dependencies."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5)))
    return ordered[index]


def iter_results(
    client: OpenAI,
    model: str,
    records: list[dict[str, Any]],
    max_tokens: int,
    concurrency: int,
    preprocess: str = "original",
) -> Iterator[dict[str, Any]]:
    """Run requests with bounded concurrency and report completed work."""
    total = len(records)
    if concurrency == 1:
        completed: Iterator[dict[str, Any]] = (
            recognize(client, model, record, max_tokens, preprocess)
            for record in records
        )
        for index, result in enumerate(completed, start=1):
            status = "ok" if result["error"] is None else "error"
            print(
                f"[{index}/{total}] {result['query_id']} {status} "
                f"{result['latency_ms']:.1f} ms",
                file=sys.stderr,
            )
            yield result
        return

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(recognize, client, model, record, max_tokens, preprocess)
            for record in records
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            status = "ok" if result["error"] is None else "error"
            print(
                f"[{index}/{total}] {result['query_id']} {status} "
                f"{result['latency_ms']:.1f} ms",
                file=sys.stderr,
            )
            yield result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL while retaining the exact path and line in diagnostics."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RunError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise RunError(f"Expected an object at {path}:{line_number}")
            records.append(record)
    return records


def index_unique(
    records: list[dict[str, Any]], source_name: str
) -> dict[str, dict[str, Any]]:
    """Index rows by query_id while rejecting missing and duplicate IDs."""
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        query_id = record.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise RunError(f"{source_name} row has no non-empty query_id")
        if query_id in indexed:
            raise RunError(f"Duplicate query_id in {source_name}: {query_id}")
        indexed[query_id] = record
    return indexed


def parse_raw_record(record: dict[str, Any]) -> dict[str, Any]:  # noqa: C901
    """Build one parsed record without modifying the raw record in memory or on disk."""
    raw_attempts = record.get("attempts")
    if raw_attempts is None:
        raw_attempts = [
            {
                "variant": record.get("preprocess", "original"),
                "raw_output": record.get("raw_output", ""),
                "latency_ms": record.get("latency_ms"),
                "finish_reason": record.get("finish_reason"),
                "usage": record.get("usage"),
                "error": record.get("error"),
            }
        ]
    if not isinstance(raw_attempts, list) or not all(
        isinstance(attempt, dict) for attempt in raw_attempts
    ):
        raise RunError(
            f"{record.get('query_id')}: attempts must be an array of objects"
        )

    parsed_attempts: list[dict[str, Any]] = []
    for attempt in raw_attempts:
        raw_output = attempt.get("raw_output", "")
        if not isinstance(raw_output, str):
            raw_output = ""
        raw_text = clean_unlimited_ocr_output(raw_output)
        parsed_attempt = {
            key: value
            for key, value in attempt.items()
            if key not in {"raw_output", "raw_text"}
        }
        parsed_attempt["raw_text"] = raw_text
        if raw_text:
            parsed_attempt["error"] = None
        elif not parsed_attempt.get("error"):
            parsed_attempt["error"] = (
                "RunError: model returned grounding markup but no OCR text"
            )
        parsed_attempts.append(parsed_attempt)

    raw_text = _merge_text(parsed_attempts)
    parsed = {
        key: value
        for key, value in record.items()
        if key not in {"attempts", "raw_output", "raw_text"}
    }
    parsed["raw_text"] = raw_text
    parsed["attempts"] = parsed_attempts
    raw_outputs: list[str] = []
    for attempt in raw_attempts:
        raw_output = attempt.get("raw_output", "")
        if isinstance(raw_output, str):
            raw_outputs.append(raw_output)
    parsed["grounding_only"] = not raw_text and any(
        "<|det|>image" in raw_output for raw_output in raw_outputs
    )
    errors: list[str] = []
    for attempt in parsed_attempts:
        error = attempt.get("error")
        if isinstance(error, str) and error and error not in errors:
            errors.append(error)
    parsed["error"] = None if raw_text else "; ".join(errors) or "no OCR text"
    return parsed


def write_jsonl(
    path: Path, records: list[dict[str, Any]], *, overwrite: bool = False
) -> None:
    """Write JSONL, refusing an existing destination unless explicitly allowed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def parse_raw_results(
    raw_path: Path, parsed_path: Path, *, overwrite: bool = False
) -> list[dict[str, Any]]:
    """Parse raw output into another file and guarantee source-path separation."""
    raw_path = raw_path.resolve()
    parsed_path = parsed_path.resolve()
    if raw_path == parsed_path or (
        parsed_path.exists() and raw_path.samefile(parsed_path)
    ):
        raise RunError("Parsed output must not overwrite the raw JSONL")
    records = read_jsonl(raw_path)
    parsed = [parse_raw_record(record) for record in records]
    write_jsonl(parsed_path, parsed, overwrite=overwrite)
    return parsed


def normalize(text: str) -> str:
    """Normalize script-preserving OCR text for phrase comparison."""
    decomposed = unicodedata.normalize("NFKD", text.casefold().replace("ё", "е"))
    without_marks = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )
    normalized = "".join(char if char.isalnum() else " " for char in without_marks)
    return " ".join(normalized.split())


def phrase_similarity(observed: str, expected: str) -> float:
    """Find the best exact or fuzzy token window for one expected phrase."""
    observed_normalized = normalize(observed)
    expected_normalized = normalize(expected)
    if not observed_normalized or not expected_normalized:
        return 0.0
    if f" {expected_normalized} " in f" {observed_normalized} ":
        return 1.0

    observed_tokens = observed_normalized.split()
    expected_tokens = expected_normalized.split()
    if len(expected_normalized.replace(" ", "")) <= 4:
        return 0.0

    best = 0.0
    target_size = len(expected_tokens)
    for window_size in range(
        max(1, target_size - 1), min(len(observed_tokens), target_size + 1) + 1
    ):
        for start in range(len(observed_tokens) - window_size + 1):
            window = " ".join(observed_tokens[start : start + window_size])
            best = max(best, SequenceMatcher(None, window, expected_normalized).ratio())
    return best


def match_threshold(value: str) -> float:
    """Use a stricter threshold for short labels to avoid accidental matches."""
    length = len(normalize(value).replace(" ", ""))
    if length <= 4:
        return 1.0
    if length <= 7:
        return 0.86
    return 0.80


def best_similarity(observed: str, gold: dict[str, Any]) -> float:
    """Score observed text against the canonical value and accepted aliases."""
    variants = [gold["value"], *gold.get("aliases", [])]
    return max(
        (
            phrase_similarity(observed, variant)
            for variant in variants
            if isinstance(variant, str) and variant
        ),
        default=0.0,
    )


def feature_result(gold: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one visible gold feature in the parsed OCR transcription."""
    raw_text = prediction.get("raw_text", "")
    if not isinstance(raw_text, str):
        raw_text = ""
    text_similarity = best_similarity(raw_text, gold)
    threshold = match_threshold(gold["value"])
    return {
        "kind": gold["kind"],
        "value": gold["value"],
        "role": gold["role"],
        "weight": ROLE_WEIGHTS[gold["role"]],
        "threshold": threshold,
        "text_similarity": round(text_similarity, 6),
        "text_match": text_similarity >= threshold,
    }


def report_percentile(values: list[float], fraction: float) -> float | None:
    """Return a nearest-rank percentile, or None when there are no values."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 3)


def ratio(numerator: float, denominator: float) -> float:
    """Return a rounded safe ratio."""
    return round(numerator / denominator, 6) if denominator else 0.0


def summarize_feature_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate hard and soft OCR feature recall with role weights."""
    total = len(rows)
    matched_text = sum(row["text_match"] for row in rows)
    weight_total = sum(row["weight"] for row in rows)
    return {
        "features": total,
        "text_recall": ratio(matched_text, total),
        "text_soft_recall": ratio(sum(row["text_similarity"] for row in rows), total),
        "text_weighted_recall": ratio(
            sum(row["weight"] for row in rows if row["text_match"]),
            weight_total,
        ),
    }


def evaluate(  # noqa: C901
    dataset: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    only_predicted: bool,
) -> dict[str, Any]:
    """Build a machine-readable evaluation report with per-query diagnostics."""
    selected = [
        row for row in dataset if not only_predicted or row["query_id"] in predictions
    ]
    details: list[dict[str, Any]] = []
    all_feature_results: list[dict[str, Any]] = []
    text_output_feature_results: list[dict[str, Any]] = []
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    latencies: list[float] = []
    errors = 0
    truncated = 0
    text_output_queries = 0
    text_output_exact_core = 0
    grounding_only_queries = 0
    request_failure_queries = 0

    for gold_row in selected:
        prediction = predictions.get(gold_row["query_id"])
        if prediction is None:
            prediction = {"raw_text": "", "error": "missing_prediction"}
        error = prediction.get("error")
        if error:
            errors += 1
        raw_text = prediction.get("raw_text", "")
        has_text = isinstance(raw_text, str) and bool(raw_text.strip())
        if has_text:
            text_output_queries += 1
        elif prediction.get("grounding_only") is True:
            grounding_only_queries += 1
        else:
            request_failure_queries += 1
        if prediction.get("finish_reason") == "length":
            truncated += 1
        latency = prediction.get("latency_ms")
        if isinstance(latency, int | float):
            latencies.append(float(latency))

        results = [
            feature_result(feature, prediction) for feature in gold_row["gold_features"]
        ]
        all_feature_results.extend(results)
        if has_text:
            text_output_feature_results.extend(results)
        for result in results:
            by_role[result["role"]].append(result)
            by_status[gold_row["catalog_status"]].append(result)
        exact_core = all(
            result["text_match"] for result in results if result["role"] != "secondary"
        )
        if has_text and exact_core:
            text_output_exact_core += 1
        details.append(
            {
                "query_id": gold_row["query_id"],
                "image_path": gold_row["image_path"],
                "catalog_status": gold_row["catalog_status"],
                "error": error,
                "latency_ms": latency,
                "finish_reason": prediction.get("finish_reason"),
                "has_text_output": has_text,
                "text_all_identity_and_disambiguation": exact_core,
                "features": results,
            }
        )

    summary = summarize_feature_results(all_feature_results)
    conditional = summarize_feature_results(text_output_feature_results)
    summary.update(
        {
            "dataset_queries": len(dataset),
            "prediction_queries": len(predictions),
            "scored_queries": len(selected),
            "coverage": ratio(len(predictions), len(dataset)),
            "errors_or_missing": errors,
            "truncated_outputs": truncated,
            "text_output_queries": text_output_queries,
            "text_output_rate": ratio(text_output_queries, len(selected)),
            "grounding_only_queries": grounding_only_queries,
            "request_failure_queries": request_failure_queries,
            "text_recall_given_output": conditional["text_recall"],
            "text_weighted_recall_given_output": conditional["text_weighted_recall"],
            "query_exact_core_recall_given_output": ratio(
                text_output_exact_core, text_output_queries
            ),
            "query_exact_core_recall": ratio(
                sum(
                    detail["text_all_identity_and_disambiguation"] for detail in details
                ),
                len(details),
            ),
            "latency_ms": {
                "mean": round(statistics.fmean(latencies), 3) if latencies else None,
                "p50": report_percentile(latencies, 0.50),
                "p95": report_percentile(latencies, 0.95),
                "max": round(max(latencies), 3) if latencies else None,
            },
        }
    )
    return {
        "summary": summary,
        "by_role": {
            role: summarize_feature_results(rows)
            for role, rows in sorted(by_role.items())
        },
        "by_catalog_status": {
            status: summarize_feature_results(rows)
            for status, rows in sorted(by_status.items())
        },
        "details": details,
    }


def write_report(path: Path, report: dict[str, Any], *, overwrite: bool) -> None:
    """Write the evaluation report under the same collision policy as parsed JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid or ambiguous unified-pipeline configurations."""
    if args.raw_input is None and not args.base_url:
        raise RunError(
            "Set OCR_BASE_URL or pass --base-url (usually http://host:port/v1)"
        )
    if args.raw_input is not None and args.output is not None:
        raise RunError("Use either --raw-input or --output, not both")
    if args.raw_input is not None and (args.limit is not None or args.query_id):
        raise RunError("--limit/--query-id apply only when calling the OCR model")
    if args.timeout <= 0 or args.max_tokens <= 0 or args.concurrency <= 0:
        raise RunError("--timeout, --max-tokens, and --concurrency must be positive")
    if args.limit is not None and args.limit <= 0:
        raise RunError("--limit must be positive")


def resolve_artifact_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Resolve and validate three distinct raw, parsed, and report paths."""
    raw_path = (args.raw_input or args.output or default_output_path()).resolve()
    parsed_path = (
        args.parsed_output or derived_output_path(raw_path, "parsed")
    ).resolve()
    report_path = (args.report or derived_output_path(raw_path, "report")).resolve()
    if len({raw_path, parsed_path, report_path}) != 3:
        raise RunError("Raw, parsed, and report paths must be different")
    if (
        parsed_path.exists()
        and report_path.exists()
        and parsed_path.samefile(report_path)
    ):
        raise RunError("Parsed and report outputs must refer to different files")
    if args.raw_input is not None:
        if not raw_path.is_file():
            raise RunError(f"Raw input does not exist: {raw_path}")
    elif raw_path.exists():
        raise RunError(
            f"Raw output already exists and is immutable: {raw_path}; choose another path"
        )
    for path in (parsed_path, report_path):
        if raw_path.exists() and path.exists() and raw_path.samefile(path):
            raise RunError(f"Derived output must not refer to raw JSONL: {path}")
        if path.exists() and not args.overwrite:
            raise RunError(f"Derived output already exists: {path}; pass --overwrite")
    return raw_path, parsed_path, report_path


def run_model(
    args: argparse.Namespace, dataset_path: Path, raw_path: Path
) -> list[dict[str, Any]]:
    """Call vLLM and stream unparsed responses to a new immutable raw JSONL."""
    records = load_dataset(dataset_path, set(args.query_id), args.limit)
    client = OpenAI(
        base_url=args.base_url.rstrip("/"),
        api_key=args.api_key,
        timeout=args.timeout,
        max_retries=0,
    )
    print(
        f"model={args.model} queries={len(records)} concurrency={args.concurrency} "
        f"preprocess={args.preprocess} prompt={UNLIMITED_OCR_PROMPT!r}",
        file=sys.stderr,
    )

    results: list[dict[str, Any]] = []
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("x", encoding="utf-8", newline="\n") as output:
        for result in iter_results(
            client,
            args.model,
            records,
            args.max_tokens,
            args.concurrency,
            args.preprocess,
        ):
            output.write(
                json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            output.flush()
            results.append(result)

    latencies = [float(result["latency_ms"]) for result in results]
    error_count = sum(result["error"] is not None for result in results)
    print(
        "ocr done "
        f"raw={raw_path} errors={error_count}/{len(results)} "
        f"latency_mean_ms={statistics.fmean(latencies):.1f} "
        f"p50_ms={percentile(latencies, 0.50):.1f} "
        f"p95_ms={percentile(latencies, 0.95):.1f}",
        file=sys.stderr,
    )
    return results


def main() -> None:
    """Run or load raw OCR, parse it separately, then evaluate in one command."""
    load_environment()
    args = parse_args()
    validate_args(args)
    dataset_path = args.dataset.resolve()
    raw_path, parsed_path, report_path = resolve_artifact_paths(args)
    dataset = read_jsonl(dataset_path)
    dataset_index = index_unique(dataset, "dataset")
    if args.raw_input is None:
        raw_rows = run_model(args, dataset_path, raw_path)
    else:
        raw_rows = read_jsonl(raw_path)
    raw_index = index_unique(raw_rows, "raw results")
    unknown_ids = set(raw_index) - set(dataset_index)
    if unknown_ids:
        raise RunError(f"Raw results contain unknown query IDs: {sorted(unknown_ids)}")

    parsed_rows = parse_raw_results(raw_path, parsed_path, overwrite=args.overwrite)
    parsed_index = index_unique(parsed_rows, "parsed results")
    report = evaluate(dataset, parsed_index, args.only_predicted)
    write_report(report_path, report, overwrite=args.overwrite)
    summary = report["summary"]
    print(
        f"parsed={parsed_path}\nreport={report_path}\n"
        f"coverage={summary['coverage']:.3f} "
        f"errors={summary['errors_or_missing']} "
        f"text_output={summary['text_output_rate']:.3f} "
        f"grounding_only={summary['grounding_only_queries']} "
        f"text_recall={summary['text_recall']:.3f} "
        f"recall_given_output={summary['text_recall_given_output']:.3f} "
        f"text_weighted={summary['text_weighted_recall']:.3f} "
        f"exact_core_queries={summary['query_exact_core_recall']:.3f}"
    )


if __name__ == "__main__":
    main()
