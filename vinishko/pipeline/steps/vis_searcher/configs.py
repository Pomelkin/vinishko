from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"


class StrictModel(BaseModel):
    """Опечатка в имени параметра конфига — ошибка, а не молча проигнорированная строка."""

    model_config = ConfigDict(extra="forbid")


class QdrantConfig(StrictModel):
    """Где лежит коллекция векторов каталога."""

    collection: str
    host: str = "localhost"
    port: int = Field(default=6333, gt=0)
    https: bool = False
    """Сервер за TLS; ключ API, если сервер его требует, читается из окружения QDRANT_API_KEY."""
    path: Path | None = None
    """Встроенный qdrant в этой директории вместо сервера, для локальной отладки; host и port тогда не читаются."""
    timeout: float = Field(default=10.0, gt=0)


class TopNSearch(StrictModel):
    """Ответ — top_k ближайших позиций как есть; если косинус лучшей ниже порога, кандидатов нет."""

    mode: Literal["top_n"]
    cosine_threshold: float = Field(ge=-1, le=1)


class GroupSearch(StrictModel):
    """Ответ по группам одинакового дизайна: позиции из top_k собираются в группы, скор группы — лучший косинус её позиций.

    Берутся top_groups лучших групп с косинусом не ниже group_threshold, кандидатами идут все их позиции, включая те,
    чьи векторы в выдачу не попали: у них скор группы. Если ни одна группа не прошла порог, кандидатов нет.
    """

    mode: Literal["groups"]
    top_groups: int = Field(gt=0)
    group_threshold: float = Field(ge=-1, le=1)


class LocalImagesConfig(StrictModel):
    """Кропы коллекции лежат в директории."""

    kind: Literal["local"]
    dir: Path


class S3ImagesConfig(StrictModel):
    """Кропы коллекции лежат в S3-совместимом хранилище; ключи — prefix/<имя из метаданных вектора>. Реквизиты — как у boto3: окружение или ~/.aws."""

    kind: Literal["s3"]
    endpoint: str
    bucket: str
    prefix: str = ""


class VisSearcherConfig(StrictModel):
    """Настройки визуального поиска."""

    model: str
    """Репозиторий Hugging Face с экспортом DinoV3ForWine: config.json, preprocess.json, model.onnx и model.bf16.onnx."""
    revision: str | None = None
    """Ревизия репозитория; без неё последняя. Коллекция помнит, какой ревизией построена, и с другой не работает."""
    device: str = "auto"
    """auto, cpu либо cuda:<индекс>; переменная окружения VIS_SEARCHER_DEV перекрывает это значение. auto — cuda:0 при доступной CUDA, иначе cpu."""
    batch_size: int = Field(default=16, gt=0)
    """Потолок батча энкодера; для TensorRT — размер профиля engine."""
    top_k: int = Field(default=10, gt=0)
    """Сколько ближайших векторов брать из коллекции."""
    qdrant: QdrantConfig
    search: Annotated[TopNSearch | GroupSearch, Field(discriminator="mode")]
    images: Annotated[LocalImagesConfig | S3ImagesConfig, Field(discriminator="kind")] | None = None
    """Кропы коллекции; None — читать оригиналы по source_image из reference_images (только запрос)."""
    reference_images: Annotated[LocalImagesConfig | S3ImagesConfig, Field(discriminator="kind")] | None = None
    """Оригинальные эталоны для NDR; если не заданы, используются кропы из images."""
    catalog_csv: Path | None = None
    """Локальные карточки Каталога для NDR v5; без CSV поля берутся из payload Qdrant."""
    cache_dir: Path = Path("~/.cache/vino")
    """Кэш engine TensorRT и картинок из S3."""
    debug_path: Path | None = None
    """Директория для разбора глазами: на каждый вызов поддиректория с меткой времени, внутри кропы запросов и картинки кандидатов."""


def load_config(path: Path | None = None) -> VisSearcherConfig:
    """Конфиг из YAML; относительные пути внутри считаются от директории файла."""
    path = (path or DEFAULT_CONFIG).resolve()
    cfg = VisSearcherConfig.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )
    base = path.parent
    cfg.cache_dir = resolve_path(cfg.cache_dir, base)
    if cfg.debug_path is not None:
        cfg.debug_path = resolve_path(cfg.debug_path, base)
    if cfg.qdrant.path is not None:
        cfg.qdrant.path = resolve_path(cfg.qdrant.path, base)
    if isinstance(cfg.images, LocalImagesConfig):
        cfg.images.dir = resolve_path(cfg.images.dir, base)
    if isinstance(cfg.reference_images, LocalImagesConfig):
        cfg.reference_images.dir = resolve_path(cfg.reference_images.dir, base)
    if cfg.catalog_csv is not None:
        cfg.catalog_csv = resolve_path(cfg.catalog_csv, base)
    return cfg


def resolve_path(p: Path, base: Path) -> Path:
    """Путь из конфига: ~ раскрывается, относительный считается от base."""
    p = p.expanduser()
    return p if p.is_absolute() else (base / p).resolve()
