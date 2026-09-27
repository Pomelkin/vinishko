"""Конфиг оркестратора: откуда брать каталог позиций. Конфиги шагов лежат рядом с шагами."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"


class StrictModel(BaseModel):
    """Опечатка в имени параметра конфига — ошибка, а не молча проигнорированная строка."""

    model_config = ConfigDict(extra="forbid")


class LocalCatalogConfig(StrictModel):
    """CSV каталога лежит на диске."""

    kind: Literal["local"]
    path: Path


class S3CatalogConfig(StrictModel):
    """CSV каталога лежит в S3-совместимом хранилище; реквизиты — как у boto3: окружение или ~/.aws."""

    kind: Literal["s3"]
    endpoint: str
    bucket: str
    key: str


class PipelineConfig(StrictModel):
    """Настройки оркестратора."""

    catalog: Annotated[LocalCatalogConfig | S3CatalogConfig, Field(discriminator="kind")]
    """CSV каталога: описание позиции в ответе API и карточки позиций второму уровню."""
    slug_column: str = "Slug"
    """Колонка CSV с идентификатором позиции; он же slug в коллекции поиска."""


def load_config(path: Path | None = None) -> PipelineConfig:
    """Конфиг из YAML; относительный путь к локальному CSV считается от директории файла."""
    path = (path or DEFAULT_CONFIG).resolve()
    cfg = PipelineConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if isinstance(cfg.catalog, LocalCatalogConfig):
        local = cfg.catalog.path.expanduser()
        cfg.catalog.path = local if local.is_absolute() else (path.parent / local).resolve()
    return cfg
