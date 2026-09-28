"""Снапшоты коллекций qdrant в директории либо в S3: снять готовую коллекцию с сервера и восстановить её на сервер, где её нет.

В хранилище на коллекцию два файла: <коллекция>.snapshot и паспорт <коллекция>.json — размер, sha256, число точек, модель, входы энкодера,
отпечаток нормализации, версия qdrant, время снятия. Паспорт пишется последним: есть паспорт — снапшот дописан. Снапшот несёт сегменты вместе
с HNSW-графом, после восстановления индекс заново не строится. qdrant_client не умеет скачивать и загружать файл снапшота, эти два запроса
идут через httpx в REST API сервера; загрузка, а не восстановление по URL, — чтобы серверу qdrant не нужен был доступ к хранилищу.
"""

import hashlib
import io
import json
import os
import shutil
import time
from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Protocol

import boto3
import httpx
from botocore.exceptions import ClientError
from kostyl.utils import setup_logger
from qdrant_client import QdrantClient
from qdrant_client.models import SnapshotDescription
from rich.console import Console
from rich.progress import BarColumn
from rich.progress import DownloadColumn
from rich.progress import Progress
from rich.progress import TextColumn
from rich.progress import TimeRemainingColumn
from rich.progress import TransferSpeedColumn

from vinishko.pipeline.steps.vis_searcher.catalog import CollectionInfo
from vinishko.pipeline.steps.vis_searcher.configs import LocalSnapshotsConfig
from vinishko.pipeline.steps.vis_searcher.configs import QdrantConfig
from vinishko.pipeline.steps.vis_searcher.configs import S3SnapshotsConfig


logger = setup_logger(fmt="detailed")
console = Console(stderr=True)
CHUNK = 1 << 20
MB = 1 << 20


def snapshot_name(collection: str) -> str:
    """Имя файла снапшота коллекции в хранилище."""
    return f"{collection}.snapshot"


def passport_name(collection: str) -> str:
    """Имя паспорта снапшота коллекции в хранилище."""
    return f"{collection}.json"


def transfer_progress() -> Progress:
    """Бар по байтам."""
    return Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def file_sha256(path: Path) -> str:
    """sha256 файла, как его считает qdrant."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ProgressFile(io.FileIO):
    """Файл на чтение: после каждого read позиция уходит в on_read. httpx читает тело multipart кусками, бар идёт за ним."""

    def __init__(self, path: Path, on_read: Callable[[int], None]) -> None:
        super().__init__(path, "rb")
        self.on_read = on_read

    def read(self, size: int | None = -1, /) -> bytes:
        chunk = super().read(size)
        self.on_read(self.tell())
        return chunk


class SnapshotStore(Protocol):
    """Снапшоты и паспорта по имени файла."""

    description: str

    def exists(self, name: str) -> bool: ...

    def get_json(self, name: str) -> dict: ...

    def put_json(self, name: str, data: dict) -> None: ...

    def fetch(self, name: str, sha256: str) -> Path:
        """Локальный путь к файлу снапшота с этим sha256; из S3 он скачивается в кэш."""
        ...

    def put(self, src: Path, name: str) -> None: ...

    def ensure_available(self) -> None: ...


class LocalSnapshots:
    """Снапшоты в директории."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.description = f"директория {root}"

    def exists(self, name: str) -> bool:
        """Есть ли файл."""
        return (self.root / name).is_file()

    def get_json(self, name: str) -> dict:
        """Прочитать JSON."""
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def put_json(self, name: str, data: dict) -> None:
        """Записать JSON."""
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def fetch(self, name: str, sha256: str) -> Path:
        """Файл лежит на диске, копировать нечего; sha256 сверяет вызывающий."""
        return self.root / name

    def put(self, src: Path, name: str) -> None:
        """Скопировать файл в директорию."""
        self.root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, self.root / name)

    def ensure_available(self) -> None:
        """Директория есть или создаётся, и в неё можно писать."""
        self.root.mkdir(parents=True, exist_ok=True)
        if not os.access(self.root, os.W_OK):
            raise PermissionError(f"нет прав на запись в {self.root}")


class S3Snapshots:
    """Снапшоты в бакете под prefix; скачанный снапшот остаётся в cache_dir и повторно не тянется, пока его sha256 совпадает с паспортом.

    Клиент boto3 с endpoint_url: реквизиты берутся как обычно, из окружения или ~/.aws. client — готовый клиент вместо boto3, для тестов.
    """

    def __init__(
        self,
        endpoint: str,
        bucket: str,
        prefix: str,
        cache_dir: Path,
        client: Any | None = None,
    ) -> None:
        self.bucket, self.prefix = bucket, prefix.strip("/")
        self.cache_dir = (
            cache_dir / bucket / self.prefix if self.prefix else cache_dir / bucket
        )
        self.client = client or boto3.client("s3", endpoint_url=endpoint)
        self.description = f"s3 {endpoint} {bucket}/{self.prefix}"

    def key(self, name: str) -> str:
        """Ключ объекта."""
        return f"{self.prefix}/{name}" if self.prefix else name

    def exists(self, name: str) -> bool:
        """Есть ли объект в бакете."""
        try:
            self.client.head_object(Bucket=self.bucket, Key=self.key(name))
        except ClientError as error:
            if error.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
        return True

    def get_json(self, name: str) -> dict:
        """JSON из бакета."""
        body = self.client.get_object(Bucket=self.bucket, Key=self.key(name))[
            "Body"
        ].read()
        return json.loads(body.decode("utf-8"))

    def put_json(self, name: str, data: dict) -> None:
        """Загрузить JSON в бакет."""
        self.client.put_object(
            Bucket=self.bucket,
            Key=self.key(name),
            Body=json.dumps(data, ensure_ascii=False, indent=1).encode("utf-8"),
            ContentType="application/json; charset=utf-8",
        )

    def fetch(self, name: str, sha256: str) -> Path:
        """Снапшот из кэша, если его sha256 совпадает, иначе скачивается из бакета в кэш."""
        cached = self.cache_dir / name
        if cached.is_file() and file_sha256(cached) == sha256:
            logger.info(f"снапшот {name} уже в кэше {cached}")
            return cached
        size = self.client.head_object(Bucket=self.bucket, Key=self.key(name))[
            "ContentLength"
        ]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with transfer_progress() as bar:
            task = bar.add_task(f"s3 → {name}", total=size)
            self.client.download_file(
                self.bucket,
                self.key(name),
                str(cached),
                Callback=lambda n: bar.advance(task, n),
            )
        return cached

    def put(self, src: Path, name: str) -> None:
        """Загрузить файл в бакет."""
        with transfer_progress() as bar:
            task = bar.add_task(f"{name} → s3", total=src.stat().st_size)
            self.client.upload_file(
                str(src),
                self.bucket,
                self.key(name),
                Callback=lambda n: bar.advance(task, n),
            )

    def ensure_available(self) -> None:
        """Бакет отвечает под текущими реквизитами; иначе ошибка boto3 как есть."""
        self.client.head_bucket(Bucket=self.bucket)


def make_snapshot_store(
    cfg: LocalSnapshotsConfig | S3SnapshotsConfig, cache_dir: Path
) -> SnapshotStore:
    """Хранилище снапшотов по конфигу."""
    if isinstance(cfg, LocalSnapshotsConfig):
        return LocalSnapshots(cfg.dir)
    return S3Snapshots(cfg.endpoint, cfg.bucket, cfg.prefix, cache_dir / "snapshots")


def server_url(cfg: QdrantConfig) -> str:
    """Адрес REST API сервера qdrant; у встроенного qdrant снапшотов нет."""
    if cfg.path is not None:
        raise RuntimeError(
            f"снапшоты снимаются и восстанавливаются только на сервере qdrant, а в конфиге встроенный ({cfg.path})"
        )
    return f"{'https' if cfg.https else 'http'}://{cfg.host}:{cfg.port}"


def auth_headers() -> dict[str, str]:
    """Ключ API из окружения, как у клиента qdrant."""
    key = os.environ.get("QDRANT_API_KEY")
    return {"api-key": key} if key else {}


def download_snapshot(
    cfg: QdrantConfig,
    collection: str,
    snapshot: SnapshotDescription,
    dest: Path,
    timeout: float,
) -> str:
    """Файл снапшота с сервера в dest; возвращает его sha256."""
    url = f"{server_url(cfg)}/collections/{collection}/snapshots/{snapshot.name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with (
        httpx.stream(
            "GET", url, headers=auth_headers(), timeout=timeout,
            trust_env=cfg.host not in {"localhost", "127.0.0.1", "::1"},
        ) as response,
        transfer_progress() as bar,
    ):
        if response.is_error:
            response.read()
            raise RuntimeError(
                f"qdrant не отдал снапшот {snapshot.name}: HTTP {response.status_code} {response.text[:500]}"
            )
        task = bar.add_task(f"qdrant → {dest.name}", total=snapshot.size)
        with dest.open("wb") as out:
            for chunk in response.iter_bytes(CHUNK):
                out.write(chunk)
                digest.update(chunk)
                bar.advance(task, len(chunk))
    return digest.hexdigest()


def upload_snapshot(
    cfg: QdrantConfig, collection: str, path: Path, sha256: str, timeout: float
) -> None:
    """Загрузить файл снапшота на сервер и дождаться восстановления коллекции; qdrant сам сверяет sha256."""
    url = f"{server_url(cfg)}/collections/{collection}/snapshots/upload"
    with transfer_progress() as bar:
        task = bar.add_task(f"{path.name} → qdrant", total=path.stat().st_size)
        with ProgressFile(path, lambda done: bar.update(task, completed=done)) as f:
            response = httpx.post(
                url,
                params={"wait": "true", "priority": "snapshot", "checksum": sha256},
                files={"snapshot": (path.name, f, "application/octet-stream")},
                headers=auth_headers(),
                timeout=timeout,
                trust_env=cfg.host not in {"localhost", "127.0.0.1", "::1"},
            )
    if response.is_error:
        raise RuntimeError(
            f"qdrant не восстановил {collection} из {path.name}: HTTP {response.status_code} {response.text[:500]}"
        )


def dump_collection(
    client: QdrantClient,
    cfg: QdrantConfig,
    info: CollectionInfo,
    store: SnapshotStore,
    work_dir: Path,
    timeout: float,
) -> dict:
    """Снапшот коллекции с сервера в хранилище: снять, скачать со сверкой sha256, удалить с сервера, положить файл, последним — паспорт."""
    collection = info.name
    version = client.info().version
    started = time.perf_counter()
    snapshot = client.create_snapshot(collection, wait=True)
    if snapshot is None:
        raise RuntimeError(f"qdrant не вернул описание снапшота {collection}")
    logger.info(
        f"qdrant {version} снял снапшот {snapshot.name}: {snapshot.size / MB:.1f} МБ, {info.count} точек"
    )
    local = work_dir / snapshot_name(collection)
    sha256 = download_snapshot(cfg, collection, snapshot, local, timeout)
    client.delete_snapshot(collection, snapshot.name, wait=True)
    if snapshot.checksum is not None and sha256 != snapshot.checksum:
        raise RuntimeError(
            f"скачанный снапшот {snapshot.name} не совпадает с сервером: sha256 {sha256}, у qdrant {snapshot.checksum}"
        )
    store.put(local, snapshot_name(collection))
    passport = {
        "collection": collection,
        "file": snapshot_name(collection),
        "size": local.stat().st_size,
        "sha256": sha256,
        "points": info.count,
        "model": info.model,
        "model_revision": info.revision,
        "encoder_input": info.encoder_input,
        "normalization": info.normalization,
        "qdrant_version": version,
        "dumped_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
    }
    store.put_json(passport_name(collection), passport)
    local.unlink()
    logger.info(
        f"снапшот {collection} лежит в {store.description}: {passport['file']} и {passport_name(collection)}, "
        f"{time.perf_counter() - started:.1f} с"
    )
    return passport


def restore_collection(
    client: QdrantClient,
    cfg: QdrantConfig,
    collection: str,
    store: SnapshotStore,
    timeout: float,
) -> None:
    """Коллекции нет в qdrant: снапшот из хранилища со сверкой sha256 загружается на сервер; нет снапшота — ошибка."""
    server_url(cfg)
    if not store.exists(passport_name(collection)):
        raise RuntimeError(
            f"в qdrant нет коллекции {collection}, и в хранилище снапшотов ({store.description}) нет {passport_name(collection)}: "
            "соберите её build_catalog либо снимите снапшот dump_collection там, где она есть"
        )
    passport = store.get_json(passport_name(collection))
    logger.info(
        f"в qdrant нет коллекции {collection}, восстанавливаю из {store.description}: снапшот {passport['dumped_at']} "
        f"с qdrant {passport['qdrant_version']} (сервер {client.info().version}), {passport['size'] / MB:.1f} МБ, "
        f"{passport['points']} точек, {passport['model']}@{passport['model_revision'][:8]}, входы {'+'.join(passport['encoder_input'])}"
    )
    started = time.perf_counter()
    path = store.fetch(passport["file"], passport["sha256"])
    sha256 = file_sha256(path)
    if sha256 != passport["sha256"]:
        raise RuntimeError(
            f"снапшот {path} не совпадает с паспортом: sha256 {sha256}, в паспорте {passport['sha256']}; снимите снапшот заново dump_collection"
        )
    upload_snapshot(cfg, collection, path, sha256, timeout)
    count = client.count(collection, exact=True).count
    if count != passport["points"]:
        raise RuntimeError(
            f"коллекция {collection} восстановлена с {count} точками, в паспорте {passport['points']}"
        )
    logger.info(
        f"коллекция {collection} восстановлена за {time.perf_counter() - started:.1f} с: {count} точек"
    )
