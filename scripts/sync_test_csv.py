"""Replace the local test manifest with the Selectel S3 object."""

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "datasets/local/test/test.csv"
DEFAULT_CREDENTIALS = Path.home() / ".codex/s3_credentials.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials-file", type=Path, default=DEFAULT_CREDENTIALS)
    args = parser.parse_args()

    credentials = json.loads(args.credentials_file.read_text(encoding="utf-8"))
    client = boto3.client(
        "s3",
        endpoint_url="https://s3.ru-6.storage.selcloud.ru",
        region_name="ru-6",
        aws_access_key_id=credentials["access_key_id"],
        aws_secret_access_key=credentials["secret_access_key"],
        config=Config(s3={"addressing_style": "path"}),
    )
    body = client.get_object(Bucket="vino", Key="data/test/test.csv")["Body"].read()
    reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
    if not {"image_filename", "slug"}.issubset(reader.fieldnames or []):
        raise ValueError("S3 CSV lacks image_filename or slug")
    rows = list(reader)
    empty = sum(not (row["slug"] or "").strip() for row in rows)
    if len(rows) != 176 or empty != 47:
        raise ValueError(f"Unexpected S3 manifest: {len(rows)} rows, {empty} empty slugs")

    if DEST.exists() and DEST.read_bytes() == body:
        print("Local test.csv already matches S3")
        return
    backup = DEST.with_name(f"test.csv.bak-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    if DEST.exists():
        shutil.copy2(DEST, backup)
    with tempfile.NamedTemporaryFile(dir=DEST.parent, prefix="test.csv.", suffix=".tmp", delete=False) as stream:
        temp = Path(stream.name)
        stream.write(body)
    try:
        os.replace(temp, DEST)
    finally:
        temp.unlink(missing_ok=True)
    print(f"Replaced {DEST} from S3; {len(rows)} rows, {empty} empty slugs")
    print(f"SHA-256: {hashlib.sha256(body).hexdigest()}")
    if backup.exists():
        print(f"Previous file: {backup}")


if __name__ == "__main__":
    main()
