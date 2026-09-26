"""Copy existing wine images inside S3 to the flat final layout, then prune data/.

Requires boto3 and scripts/.upload_s3_credentials.json. Run without --apply to
inspect the plan, then run with --apply to copy images, upload two CSV files,
and remove all other objects under the selected data/ prefix. No image bytes
are uploaded from this machine.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PLAN = DATA / "technical/s3_image_copy_plan.csv"
CREDENTIALS = Path(__file__).with_name(".upload_s3_credentials.json")
ENDPOINT = "https://s3.ru-6.storage.selcloud.ru"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_plan() -> tuple[list[tuple[str, str]], set[str]]:
    catalog = read_csv(DATA / "catalog/catalog.csv")
    tests = read_csv(DATA / "test/test.csv")
    plan_rows = read_csv(PLAN)
    expected_targets = {
        "data/catalog/images/" + row["image_filename"]
        for row in catalog if row["image_filename"]
    } | {
        "data/test/images/" + row["image_filename"]
        for row in tests
    }
    expected_targets.update({"data/catalog/catalog.csv", "data/test/test.csv"})
    copies: list[tuple[str, str]] = []
    for row in plan_rows:
        source, target = row["source_key"], row["target_key"]
        if not source.startswith(("data/strapi/img/", "data/test_merged/")):
            raise ValueError(f"unexpected source key: {source}")
        if not target.startswith(("data/catalog/images/", "data/test/images/")):
            raise ValueError(f"unexpected destination key: {target}")
        if ".." in Path(source).parts or ".." in Path(target).parts:
            raise ValueError("parent path in S3 key")
        copies.append((source, target))
    if len(copies) != 2231 or len({target for _, target in copies}) != len(copies):
        raise ValueError("unexpected or duplicate image copies")
    if {target for _, target in copies} != expected_targets - {
        "data/catalog/catalog.csv", "data/test/test.csv"
    }:
        raise ValueError("S3 copy plan differs from final CSV image names")
    for dataset, rows in [("catalog", catalog), ("test", tests)]:
        for row in rows:
            name = row["image_filename"]
            if name and ("/" in name or "\\" in name or not (DATA / dataset / "images" / name).is_file()):
                raise ValueError(f"missing flat local image: {dataset}/{name}")
    return copies, expected_targets


def list_objects(client, bucket: str, root: str) -> set[str]:
    pages = client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=root)
    return {item["Key"] for page in pages for item in page.get("Contents", [])}


def delete_extras(client, bucket: str, extras: list[str], workers: int) -> int:
    batches = [extras[start:start + 1000] for start in range(0, len(extras), 1000)]

    def delete_batch(batch: list[str]) -> int:
        response = client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
        )
        return len(response.get("Errors", []))

    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(delete_batch, batch) for batch in batches]
        for future in as_completed(futures):
            errors += future.result()
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="perform S3 writes and cleanup")
    parser.add_argument("--cleanup-only", action="store_true", help="delete obsolete keys after final objects exist")
    parser.add_argument("--bucket", default="vino")
    parser.add_argument("--prefix", default="", help="optional prefix before data/")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    prefix = args.prefix.strip("/")
    if ".." in prefix.split("/"):
        raise ValueError("invalid prefix")
    prefix = f"{prefix}/" if prefix else ""
    scoped_root = prefix + "data/"

    copies, expected_relative = validate_plan()
    credentials = json.loads(CREDENTIALS.read_text(encoding="utf-8"))
    client = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        region_name="ru-6",
        aws_access_key_id=credentials["access_key_id"],
        aws_secret_access_key=credentials["secret_access_key"],
        config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 10, "mode": "standard"}),
    )
    old_keys = list_objects(client, args.bucket, scoped_root)
    expected = {prefix + key for key in expected_relative}
    if old_keys == expected:
        print(f"Already complete: {len(expected)} final objects under {scoped_root}")
        return
    if args.cleanup_only:
        if expected - old_keys:
            raise ValueError(f"{len(expected - old_keys)} final objects are missing; cleanup cancelled")
        extras = sorted(old_keys - expected)
        print(f"Cleanup plan: {len(extras)} obsolete objects under {scoped_root}", flush=True)
        if args.apply:
            errors = delete_extras(client, args.bucket, extras, args.workers)
            final = list_objects(client, args.bucket, scoped_root)
            if final != expected:
                raise ValueError(f"S3 final inventory differs: missing={len(expected-final)}, extra={len(final-expected)}, delete_errors={errors}")
            print(f"Done: {len(final)} final objects; removed {len(extras)} old objects", flush=True)
        return
    source_keys = {prefix + source for source, _ in copies}
    missing = source_keys - old_keys
    if missing:
        raise ValueError(f"{len(missing)} source images are missing on S3; cleanup cancelled")
    print(f"S3 plan: {len(copies)} server-side image copies, 2 CSV uploads, "
          f"{len(old_keys - expected)} old objects to remove under {scoped_root}")
    if not args.apply:
        return

    def copy_one(pair: tuple[str, str]) -> None:
        source, target = pair
        client.copy_object(
            Bucket=args.bucket,
            CopySource={"Bucket": args.bucket, "Key": prefix + source},
            Key=prefix + target,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(copy_one, pair) for pair in copies]
        for future in as_completed(futures):
            future.result()
    print(f"Copied {len(copies)} images inside S3")

    for dataset in ("catalog", "test"):
        path = DATA / dataset / f"{dataset}.csv"
        client.put_object(Bucket=args.bucket, Key=prefix + f"data/{dataset}/{dataset}.csv",
                          Body=path.read_bytes(), ContentType="text/csv; charset=utf-8")
    print("Uploaded catalog.csv and test.csv")

    current = list_objects(client, args.bucket, scoped_root)
    if expected - current:
        raise ValueError(f"{len(expected - current)} final objects are missing; cleanup cancelled")
    extras = sorted(current - expected)
    errors = delete_extras(client, args.bucket, extras, args.workers)
    final = list_objects(client, args.bucket, scoped_root)
    if final != expected:
        raise ValueError(f"S3 final inventory differs: missing={len(expected-final)}, extra={len(final-expected)}, delete_errors={errors}")
    print(f"Done: {len(final)} final objects under {scoped_root}; removed {len(extras)} old objects")


if __name__ == "__main__":
    main()
