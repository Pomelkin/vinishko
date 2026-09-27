"""Каталог позиций: строки CSV по slug из локального файла либо из S3. Описание позиции для ответа API и карточки второму уровню."""

import csv
from dataclasses import dataclass
from io import StringIO

import boto3

from vinishko.pipeline.configs import LocalCatalogConfig, S3CatalogConfig


@dataclass(frozen=True, slots=True)
class Catalog:
    """Строки каталога по slug: все колонки CSV как строки, пробелы по краям сняты."""

    rows: dict[str, dict[str, str]]
    description: str
    """Откуда прочитан: путь к файлу либо адрес в S3."""

    def get(self, slug: str) -> dict[str, str]:
        """Строка позиции; slug, которого нет в каталоге, — ошибка: коллекция и каталог должны быть из одной выгрузки."""
        if slug not in self.rows:
            raise KeyError(f"в каталоге ({self.description}) нет позиции {slug}")
        return self.rows[slug]

    def __contains__(self, slug: str) -> bool:
        return slug in self.rows

    def __len__(self) -> int:
        return len(self.rows)


def read_catalog_text(cfg: LocalCatalogConfig | S3CatalogConfig) -> tuple[str, str]:
    """Текст CSV и описание источника."""
    if isinstance(cfg, LocalCatalogConfig):
        return cfg.path.read_text(encoding="utf-8-sig"), f"файл {cfg.path}"
    body = boto3.client("s3", endpoint_url=cfg.endpoint).get_object(Bucket=cfg.bucket, Key=cfg.key)["Body"].read()
    return body.decode("utf-8-sig"), f"s3 {cfg.endpoint} {cfg.bucket}/{cfg.key}"


def load_catalog(cfg: LocalCatalogConfig | S3CatalogConfig, slug_column: str = "Slug") -> Catalog:
    """Каталог из CSV по конфигу; пустой или повторный slug — ошибка."""
    text, description = read_catalog_text(cfg)
    reader = csv.DictReader(StringIO(text))
    columns = reader.fieldnames or []
    if slug_column not in columns:
        raise RuntimeError(f"в каталоге ({description}) нет колонки {slug_column!r}; есть {columns[:8]}")
    rows: dict[str, dict[str, str]] = {}
    for record in reader:
        slug = (record[slug_column] or "").strip()
        if not slug:
            raise RuntimeError(f"в каталоге ({description}) строка с пустым {slug_column}")
        if slug in rows:
            raise RuntimeError(f"в каталоге ({description}) slug {slug} повторяется")
        rows[slug] = {key: (value or "").strip() for key, value in record.items() if key is not None}
    return Catalog(rows, description)
