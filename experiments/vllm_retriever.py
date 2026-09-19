from __future__ import annotations

import asyncio
import base64
import json
import math
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI
from tqdm.auto import tqdm


# ============================================================
# SETTINGS
# ============================================================

VLLM_BASE_URL = "https://f60b-136-85-48-58.ngrok-free.app/v1"
VLLM_API_KEY = "EMPTY"
# The served model must support two images in one chat request.
MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"

DATA_DIR = Path(__file__).with_name("data")
CATALOG_CSV = DATA_DIR / "catalog_dataset.csv"
CATALOG_IMAGES_DIR = DATA_DIR / "images_without_garbarage"

# Replace these paths/column names when the new labeled eval dataset arrives.
EVAL_CSV = DATA_DIR / "eval" / "eval_dataset.csv"
EVAL_IMAGES_DIR = DATA_DIR / "eval" / "images"
EVAL_ID_COLUMN = "query_id"
EVAL_IMAGE_COLUMN = "image_path"

CATALOG_ID_COLUMN = "Slug"
CATALOG_IMAGE_COLUMN = "Название фото"
CATALOG_TEXT_COLUMNS = (
    "Название вина",
    "Винодельня",
    "Категория",
    "Цвет",
    "Сорт винограда",
    "Винтаж",
    "Сахар",
    "Игристое",
    "Креплёное",
    "Безалкогольное",
)

TOP_N = 10
MAX_CONCURRENCY = 10
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 600.0

RESULT_COLUMN = "retrieved_products"
RESULT_ERROR_COLUMN = "retrieval_error"
OUTPUT_CSV = DATA_DIR / "eval_with_retrieved_products.csv"

# Set an integer for a quick partial run or leave None for the full dataset.
CATALOG_LIMIT: int | None = None
EVAL_LIMIT: int | None = None

SYSTEM_PROMPT = """
You are a highly precise wine product matcher.

You will receive two images:
1. QUERY PHOTO: a real-world photo of a wine bottle or label. It may contain
   glare, blur, perspective distortion, partial occlusion, a difficult
   background, or poor lighting.
2. CATALOG CANDIDATE: a clean catalog image, accompanied by product metadata.

Decide whether both images represent the same exact wine product. Focus on the
label: producer and brand, product or cuvee name, series, grape variety, wine
color, sweetness, sparkling status, vintage, reserve designation, typography,
emblems, and layout. Ignore photographic conditions and small packaging design
changes that do not change product identity.

Closely related wines are not the same product. Return 0 for another wine from
the same producer or series, especially when the grape, color, sweetness,
sparkling status, vintage, or reserve designation differs.

Do not explain your reasoning. Your entire response MUST be exactly one
character: 0 or 1.
""".strip()

VLLM_EXTRA_BODY: dict[str, Any] = {
    "top_k": 2,
    "chat_template_kwargs": {
        "enable_thinking": False,
    },
}


def image_to_data_url(image_path: Path) -> str:
    """Encode a local image for an OpenAI-compatible multimodal request."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    media_type = mimetypes.guess_type(image_path.name)[0]
    if media_type is None and image_path.suffix.lower() == ".webp":
        media_type = "image/webp"
    if media_type is None or not media_type.startswith("image/"):
        raise ValueError(f"Unsupported image type: {image_path}")

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def build_user_content(
    query_image_url: str,
    candidate_image_url: str,
    product_text: str,
) -> list[dict[str, Any]]:
    return [
        {
            "type": "text",
            "text": "QUERY PHOTO (image 1):",
        },
        {
            "type": "image_url",
            "image_url": {"url": query_image_url},
        },
        {
            "type": "text",
            "text": f"CATALOG CANDIDATE (image 2):\n{product_text}",
        },
        {
            "type": "image_url",
            "image_url": {"url": candidate_image_url},
        },
        {
            "type": "text",
            "text": "Return exactly one character: 0 or 1.",
        },
    ]


def normalize_binary_token(token: str | None) -> str | None:
    if token is None:
        return None

    normalized = token.strip()
    return normalized if normalized in {"0", "1"} else None


def extract_binary_probabilities(response: Any) -> dict[str, Any]:
    """Extract and normalize first-token probabilities within {0, 1}."""
    choice = response.choices[0]
    p0_raw = 0.0
    p1_raw = 0.0
    top_tokens: list[dict[str, Any]] = []

    if choice.logprobs is not None and choice.logprobs.content:
        first_position = choice.logprobs.content[0]
        candidates = [first_position, *(first_position.top_logprobs or [])]

        for item in candidates:
            probability = math.exp(item.logprob)
            label = normalize_binary_token(item.token)
            top_tokens.append(
                {
                    "token": repr(item.token),
                    "logprob": item.logprob,
                    "probability": probability,
                }
            )

            if label == "0":
                p0_raw = max(p0_raw, probability)
            elif label == "1":
                p1_raw = max(p1_raw, probability)

    denominator = p0_raw + p1_raw

    return {
        "output": choice.message.content,
        "p0_raw": p0_raw,
        "p1_raw": p1_raw,
        "p0": p0_raw / denominator if denominator else None,
        "p1": p1_raw / denominator if denominator else None,
        "top_tokens": top_tokens,
    }


def to_python_scalar(value: Any) -> Any:
    """Convert numpy/pandas scalars into JSON-serializable Python values."""
    return value.item() if hasattr(value, "item") else value


def format_product(row: pd.Series) -> str:
    values = []
    for column in CATALOG_TEXT_COLUMNS:
        value = row[column]
        if pd.notna(value) and str(value).strip():
            value = to_python_scalar(value)
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            values.append(f"{column}: {value}")
    return "\n".join(values)


def load_catalog() -> pd.DataFrame:
    catalog = pd.read_csv(CATALOG_CSV)
    required = {
        CATALOG_ID_COLUMN,
        CATALOG_IMAGE_COLUMN,
        *CATALOG_TEXT_COLUMNS,
    }
    missing_columns = required.difference(catalog.columns)
    if missing_columns:
        raise ValueError(f"Catalog is missing columns: {sorted(missing_columns)}")

    catalog = catalog.dropna(
        subset=[CATALOG_ID_COLUMN, CATALOG_IMAGE_COLUMN]
    ).copy()
    catalog["id"] = catalog[CATALOG_ID_COLUMN].astype(str)
    catalog["image_path"] = catalog[CATALOG_IMAGE_COLUMN].map(
        lambda name: CATALOG_IMAGES_DIR / str(name)
    )
    catalog["text"] = catalog.apply(format_product, axis=1)

    missing_images = [path for path in catalog["image_path"] if not path.is_file()]
    if missing_images:
        examples = ", ".join(str(path) for path in missing_images[:3])
        raise FileNotFoundError(
            f"Catalog has {len(missing_images)} missing images. First: {examples}"
        )

    if CATALOG_LIMIT is not None:
        catalog = catalog.head(CATALOG_LIMIT).copy()

    return catalog[["id", "image_path", "text"]].reset_index(drop=True)


def load_eval_dataset() -> pd.DataFrame:
    if not EVAL_CSV.is_file():
        raise FileNotFoundError(
            f"Eval dataset is not ready yet. Put its CSV at {EVAL_CSV} "
            f"and images at {EVAL_IMAGES_DIR}."
        )

    separator = "\t" if EVAL_CSV.suffix.lower() == ".tsv" else ","
    eval_df = pd.read_csv(EVAL_CSV, sep=separator)
    required = {EVAL_ID_COLUMN, EVAL_IMAGE_COLUMN}
    missing_columns = required.difference(eval_df.columns)
    if missing_columns:
        raise ValueError(f"Eval dataset is missing columns: {sorted(missing_columns)}")

    eval_df["query_image_path"] = eval_df[EVAL_IMAGE_COLUMN].map(
        lambda name: EVAL_IMAGES_DIR / str(name)
    )
    missing_images = [
        path for path in eval_df["query_image_path"] if not path.is_file()
    ]
    if missing_images:
        examples = ", ".join(str(path) for path in missing_images[:3])
        raise FileNotFoundError(
            f"Eval dataset has {len(missing_images)} missing images. First: {examples}"
        )

    if EVAL_LIMIT is not None:
        eval_df = eval_df.head(EVAL_LIMIT).copy()

    return eval_df


@dataclass(slots=True)
class CandidateScore:
    position: int
    product_slug: Any
    output: str | None
    p0_raw: float | None
    p1_raw: float | None
    p0: float | None
    p1: float | None
    top_tokens: list[dict[str, Any]] | None
    error: str | None


class VLLMWineRetriever:
    """Score every catalog product and return the TOP_N most likely matches."""

    def __init__(
        self,
        *,
        catalog: pd.DataFrame,
        client: AsyncOpenAI,
        model: str,
        top_n: int,
        max_concurrency: int,
        max_retries: int,
        retry_base_delay_seconds: float,
        system_prompt: str,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        missing = {"id", "image_path", "text"}.difference(catalog.columns)
        if missing:
            raise ValueError(f"catalog: missing columns {sorted(missing)}")
        if catalog["id"].duplicated().any():
            raise ValueError("catalog: product slugs must be unique")
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")

        self.catalog = catalog[["id", "image_path", "text"]].reset_index(
            drop=True
        ).copy()
        self.client = client
        self.model = model
        self.top_n = top_n
        self.max_retries = max_retries
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.system_prompt = system_prompt
        self.extra_body = extra_body or {}
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.last_scores: pd.DataFrame | None = None

    async def _score_one(
        self,
        *,
        position: int,
        product_slug: Any,
        product_text: str,
        candidate_image_path: Path,
        query_image_url: str,
    ) -> CandidateScore:
        last_error: str | None = None

        for attempt in range(self.max_retries):
            try:
                candidate_image_url = image_to_data_url(candidate_image_path)
                async with self._semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {
                                "role": "user",
                                "content": build_user_content(
                                    query_image_url,
                                    candidate_image_url,
                                    product_text,
                                ),
                            },
                        ],
                        temperature=0,
                        max_tokens=1,
                        logprobs=True,
                        top_logprobs=5,
                        extra_body=self.extra_body,
                    )

                probabilities = extract_binary_probabilities(response)
                if probabilities["p1"] is None:
                    raise ValueError(
                        "The first-token logprobs contain neither 0 nor 1"
                    )

                return CandidateScore(
                    position=position,
                    product_slug=to_python_scalar(product_slug),
                    output=probabilities["output"],
                    p0_raw=probabilities["p0_raw"],
                    p1_raw=probabilities["p1_raw"],
                    p0=probabilities["p0"],
                    p1=probabilities["p1"],
                    top_tokens=probabilities["top_tokens"],
                    error=None,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                tqdm.write(
                    f"[candidate={product_slug}] attempt "
                    f"{attempt + 1}/{self.max_retries}: {last_error}"
                )
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(
                        self.retry_base_delay_seconds * (2**attempt)
                    )

        return CandidateScore(
            position=position,
            product_slug=to_python_scalar(product_slug),
            output=None,
            p0_raw=None,
            p1_raw=None,
            p0=None,
            p1=None,
            top_tokens=None,
            error=last_error,
        )

    async def score_catalog(self, query_image: str | Path) -> pd.DataFrame:
        """Calculate a match score for every catalog product."""
        query_image_path = Path(query_image)
        query_image_url = image_to_data_url(query_image_path)
        tasks = [
            asyncio.create_task(
                self._score_one(
                    position=position,
                    product_slug=row.id,
                    product_text=str(row.text),
                    candidate_image_path=Path(row.image_path),
                    query_image_url=query_image_url,
                )
            )
            for position, row in enumerate(self.catalog.itertuples(index=False))
        ]

        results: list[CandidateScore] = []
        progress = tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Candidates: {query_image_path.name}",
            leave=False,
        )
        try:
            for completed_task in progress:
                results.append(await completed_task)
        finally:
            progress.close()

        scores = pd.DataFrame(
            {
                "position": item.position,
                "slug": item.product_slug,
                "output": item.output,
                "p_0_raw": item.p0_raw,
                "p_1_raw": item.p1_raw,
                "p_0": item.p0,
                "p_1": item.p1,
                "top_tokens": item.top_tokens,
                "error": item.error,
            }
            for item in results
        ).sort_values("position", kind="stable").reset_index(drop=True)

        self.last_scores = scores
        return scores

    async def retrieve(self, query_image: str | Path) -> list[dict[str, Any]]:
        """Return TOP_N catalog products ordered by descending match score."""
        scores = await self.score_catalog(query_image)
        successful = scores.dropna(subset=["p_1"])

        if successful.empty:
            errors = scores["error"].dropna().head(3).tolist()
            raise RuntimeError(
                "Could not score any catalog product. "
                f"First errors: {errors}"
            )

        top_products = (
            successful.sort_values("p_1", ascending=False, kind="stable")
            .head(self.top_n)[["slug", "p_1"]]
            .to_dict("records")
        )
        return [
            {
                "slug": to_python_scalar(product["slug"]),
                "p_1": float(product["p_1"]),
            }
            for product in top_products
        ]


async def apply_retriever_async(
    dataframe: pd.DataFrame,
    *,
    retriever: VLLMWineRetriever,
    query_image_column: str,
    result_column: str = "retrieved_products",
    error_column: str = "retrieval_error",
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Retrieve catalog candidates for each query image and save the result."""
    if query_image_column not in dataframe.columns:
        raise ValueError(f"Missing query image column: {query_image_column!r}")

    result = dataframe.copy()
    result[result_column] = pd.Series(pd.NA, index=result.index, dtype="string")
    result[error_column] = pd.Series(pd.NA, index=result.index, dtype="string")

    progress = tqdm(result.index, total=len(result), desc="Query images")
    try:
        for index in progress:
            query_image = Path(result.at[index, query_image_column])

            try:
                retrieved_products = await retriever.retrieve(query_image)
                result.at[index, result_column] = json.dumps(
                    retrieved_products,
                    ensure_ascii=False,
                )
                result.at[index, error_column] = pd.NA
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                result.at[index, error_column] = error
                tqdm.write(f"[query={query_image}] {error}")
    finally:
        progress.close()

    if output_csv is not None:
        result.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return result


async def main() -> pd.DataFrame:
    catalog = load_catalog()
    # eval_df = load_eval_dataset()

    client = AsyncOpenAI(
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        timeout=REQUEST_TIMEOUT_SECONDS,
        # Retries are implemented explicitly with asyncio.sleep.
        max_retries=0,
    )
    retriever = VLLMWineRetriever(
        catalog=catalog,
        client=client,
        model=MODEL,
        top_n=TOP_N,
        max_concurrency=MAX_CONCURRENCY,
        max_retries=MAX_RETRIES,
        retry_base_delay_seconds=RETRY_BASE_DELAY_SECONDS,
        system_prompt=SYSTEM_PROMPT,
        extra_body=VLLM_EXTRA_BODY,
    )

    print(await retriever.retrieve(r"E:\Pycharm Projects\vinishko\experiments\photo_viewer\sorted_photos\abrau-dyurso-russkoe-igristoe-polusladkoe-shardone-beloe-12\images.jpg"))

    # result_df = await apply_retriever_async(
    #     eval_df,
    #     retriever=retriever,
    #     query_image_column="query_image_path",
    #     result_column=RESULT_COLUMN,
    #     error_column=RESULT_ERROR_COLUMN,
    #     output_csv=OUTPUT_CSV,
    # )

    # print(f"Saved {len(result_df)} rows to {OUTPUT_CSV}")
    # return result_df


if __name__ == "__main__":
    asyncio.run(main())
