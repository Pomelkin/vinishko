"""Каталог позиций: строки CSV по slug из локального файла либо из S3. Описание позиции для ответа API и карточки второму уровню."""

import csv
from dataclasses import dataclass
from io import StringIO

import boto3
from botocore.exceptions import BotoCoreError
from botocore.exceptions import ClientError
from botocore.exceptions import NoCredentialsError
from kostyl.utils import setup_logger

from vinishko.pred.pipeline.configs import LocalCatalogConfig
from vinishko.pred.pipeline.configs import S3CatalogConfig


logger = setup_logger(fmt="detailed")
S3_REASONS = {
    "NoSuchKey": "объекта нет в бакете",
    "NoSuchBucket": "бакета нет",
    "AccessDenied": "нет доступа: у реквизитов нет прав на объект, либо объекта нет, а прав на список бакета нет",
    "InvalidAccessKeyId": "ключ доступа неизвестен хранилищу",
    "SignatureDoesNotMatch": "секретный ключ не подходит к ключу доступа",
}
"""Коды ошибок S3 при чтении каталога → причина для сообщения; остальные коды идут как есть."""


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
    """Текст CSV и описание источника; нет файла либо объекта, нет доступа или связи с хранилищем — ошибка с адресом каталога и причиной."""
    if isinstance(cfg, LocalCatalogConfig):
        description = f"файл {cfg.path}"
        if not cfg.path.is_file():
            raise RuntimeError(f"каталог недоступен ({description}): файла нет")
        return cfg.path.read_text(encoding="utf-8-sig"), description
    description = f"s3 {cfg.endpoint} {cfg.bucket}/{cfg.key}"
    try:
        body = (
            boto3.client("s3", endpoint_url=cfg.endpoint)
            .get_object(Bucket=cfg.bucket, Key=cfg.key)["Body"]
            .read()
        )
    except ClientError as error:
        code = error.response["Error"]["Code"]
        reason = S3_REASONS.get(
            code, f"{code} {error.response['Error'].get('Message', '')}".strip()
        )
        raise RuntimeError(f"каталог недоступен ({description}): {reason}") from error
    except NoCredentialsError as error:
        raise RuntimeError(
            f"каталог недоступен ({description}): нет реквизитов S3, задайте AWS_ACCESS_KEY_ID и AWS_SECRET_ACCESS_KEY в окружении либо ~/.aws"
        ) from error
    # нет связи с endpoint, таймаут и прочее до ответа хранилища
    except BotoCoreError as error:
        raise RuntimeError(f"каталог недоступен ({description}): {error}") from error
    return body.decode("utf-8-sig"), description


def load_catalog(
    cfg: LocalCatalogConfig | S3CatalogConfig, slug_column: str = "Slug"
) -> Catalog:
    """Каталог из CSV по конфигу; пустой или повторный slug — ошибка."""
    text, description = read_catalog_text(cfg)
    reader = csv.DictReader(StringIO(text))
    columns = reader.fieldnames or []
    if slug_column not in columns:
        raise RuntimeError(
            f"в каталоге ({description}) нет колонки {slug_column!r}; есть {columns[:8]}"
        )
    rows: dict[str, dict[str, str]] = {}
    for record in reader:
        slug = (record[slug_column] or "").strip()
        if not slug:
            raise RuntimeError(
                f"в каталоге ({description}) строка с пустым {slug_column}"
            )
        if slug in rows:
            raise RuntimeError(f"в каталоге ({description}) slug {slug} повторяется")
        rows[slug] = {
            key: (value or "").strip()
            for key, value in record.items()
            if key is not None
        }
    logger.info(f"каталог: {description}, позиций {len(rows)}")
    return Catalog(rows, description)
