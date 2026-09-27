"""Хранилище кропов коллекции: директория либо S3-совместимый бакет с кэшем на диске. Имя файла — из метаданных вектора.

Рядом с картинками лежит manifest.json последней сборки: чем нормализованы кропы и какая коллекция их писала."""

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

import boto3
import numpy as np
from botocore.exceptions import ClientError
from PIL import Image

from vinishko.pipeline.steps.vis_searcher.configs import (
    LocalImagesConfig,
    S3ImagesConfig,
)

CONTENT_TYPES = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
PIL_FORMATS = {"jpg": "JPEG", "png": "PNG", "webp": "WEBP"}
MANIFEST = "manifest.json"
"""Паспорт хранилища: `normalization` — crop_config нормализатора при сборке, `fingerprint` — его отпечаток, тот же лежит в метаданных точек коллекции,
`collection`, модель, `encoder_input`, формат картинок, `built_at`. Поиск при старте сверяет с ним и коллекцию, и нормализатор запроса."""


def encode_image(image: np.ndarray, fmt: str, quality: int) -> bytes:
    """uint8 RGB → байты файла."""
    buf = BytesIO()
    Image.fromarray(image).save(buf, PIL_FORMATS[fmt], quality=quality)
    return buf.getvalue()


def decode_image(data: bytes) -> np.ndarray:
    """Байты файла → uint8 RGB."""
    return np.asarray(Image.open(BytesIO(data)).convert("RGB"))


class ImageStore(Protocol):
    """Кропы коллекции по имени файла."""

    description: str

    def put(self, name: str, image: np.ndarray, fmt: str, quality: int) -> None: ...

    def get(self, name: str) -> np.ndarray: ...

    def put_json(self, name: str, data: dict) -> None: ...

    def get_json(self, name: str) -> dict: ...

    def exists(self, name: str) -> bool: ...

    def ensure_available(self) -> None: ...


def read_manifest(store: ImageStore) -> dict:
    """Манифест хранилища; без него не проверить, чем нормализованы картинки, и это ошибка."""
    if not store.exists(MANIFEST):
        raise RuntimeError(
            f"в хранилище картинок ({store.description}) нет {MANIFEST}: картинки собраны до его появления, пересоберите коллекцию build_catalog"
        )
    return store.get_json(MANIFEST)


class LocalStore:
    """Кропы в директории."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.description = f"директория {root}"

    def put(self, name: str, image: np.ndarray, fmt: str, quality: int) -> None:
        """Записать кроп."""
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / name).write_bytes(encode_image(image, fmt, quality))

    def get(self, name: str) -> np.ndarray:
        """Прочитать кроп."""
        return decode_image((self.root / name).read_bytes())

    def put_json(self, name: str, data: dict) -> None:
        """Записать JSON."""
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def get_json(self, name: str) -> dict:
        """Прочитать JSON."""
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def exists(self, name: str) -> bool:
        """Есть ли файл."""
        return (self.root / name).is_file()

    def ensure_available(self) -> None:
        """Директория есть или создаётся, и в неё можно писать."""
        self.root.mkdir(parents=True, exist_ok=True)
        if not os.access(self.root, os.W_OK):
            raise PermissionError(f"нет прав на запись в {self.root}")


class S3Store:
    """Кропы в бакете под prefix; прочитанное и записанное оседает в cache_dir, повторно из сети не тянется.

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
        self.description = f"s3 {endpoint} {bucket}/{self.prefix}, кэш {self.cache_dir}"

    def key(self, name: str) -> str:
        """Ключ объекта."""
        return f"{self.prefix}/{name}" if self.prefix else name

    def put(self, name: str, image: np.ndarray, fmt: str, quality: int) -> None:
        """Загрузить кроп в бакет и положить копию в кэш."""
        data = encode_image(image, fmt, quality)
        self.client.put_object(
            Bucket=self.bucket,
            Key=self.key(name),
            Body=data,
            ContentType=CONTENT_TYPES[fmt],
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / name).write_bytes(data)

    def get(self, name: str) -> np.ndarray:
        """Кроп из кэша, иначе из бакета с записью в кэш."""
        cached = self.cache_dir / name
        if not cached.is_file():
            data = self.client.get_object(Bucket=self.bucket, Key=self.key(name))[
                "Body"
            ].read()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(data)
        return decode_image(cached.read_bytes())

    def put_json(self, name: str, data: dict) -> None:
        """Загрузить JSON в бакет; в кэш не кладётся."""
        self.client.put_object(
            Bucket=self.bucket,
            Key=self.key(name),
            Body=json.dumps(data, ensure_ascii=False, indent=1).encode("utf-8"),
            ContentType="application/json; charset=utf-8",
        )

    def get_json(self, name: str) -> dict:
        """JSON из бакета мимо кэша: манифест нужен свежий, другая сборка могла его переписать."""
        body = self.client.get_object(Bucket=self.bucket, Key=self.key(name))[
            "Body"
        ].read()
        return json.loads(body.decode("utf-8"))

    def exists(self, name: str) -> bool:
        """Есть ли объект в бакете."""
        try:
            self.client.head_object(Bucket=self.bucket, Key=self.key(name))
        except ClientError as error:
            if error.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
        return True

    def ensure_available(self) -> None:
        """Бакет отвечает под текущими реквизитами; иначе ошибка boto3 как есть."""
        self.client.head_bucket(Bucket=self.bucket)


def make_store(cfg: LocalImagesConfig | S3ImagesConfig, cache_dir: Path) -> ImageStore:
    """Хранилище по конфигу."""
    if isinstance(cfg, LocalImagesConfig):
        return LocalStore(cfg.dir)
    return S3Store(cfg.endpoint, cfg.bucket, cfg.prefix, cache_dir / "images")
