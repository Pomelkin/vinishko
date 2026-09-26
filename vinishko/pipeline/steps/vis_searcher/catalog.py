"""Коллекция векторов каталога в qdrant: схема точки, подключение, создание, поиск и проверки."""

import uuid
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams

from vinishko.pipeline.steps.vis_searcher.configs import QdrantConfig

POINT_NAMESPACE = uuid.UUID("3b1e9c6a-5d2f-4e8b-9a7c-1f0d2e3c4b5a")
"""Идентификатор точки — uuid5 от slug: повторная сборка перезаписывает точку, а не плодит дубли."""
FIELD_SLUG, FIELD_GROUP, FIELD_GROUP_SLUGS, FIELD_IMAGE = (
    "slug",
    "group",
    "group_slugs",
    "image",
)
FIELD_MODEL, FIELD_REVISION, FIELD_INPUT_SIZE = "model", "model_revision", "input_size"
"""Чем построена коллекция; лежит в каждой точке, поиск сверяет с загруженной моделью."""


def point_id(slug: str) -> str:
    """Идентификатор точки по slug."""
    return str(uuid.uuid5(POINT_NAMESPACE, slug))


def connect(cfg: QdrantConfig) -> QdrantClient:
    """Клиент qdrant: встроенный по пути либо сервер."""
    if cfg.path is not None:
        return QdrantClient(path=str(cfg.path))
    return QdrantClient(host=cfg.host, port=cfg.port, timeout=int(cfg.timeout))


def create_collection(client: QdrantClient, name: str, size: int) -> None:
    """Новая коллекция под L2-нормированные векторы, метрика — косинус."""
    client.create_collection(
        name, vectors_config=VectorParams(size=size, distance=Distance.COSINE)
    )


def vector_size(client: QdrantClient, name: str) -> int:
    """Размерность векторов коллекции."""
    vectors = client.get_collection(name).config.params.vectors
    if not isinstance(vectors, VectorParams):
        raise RuntimeError(
            f"коллекция {name} с именованными векторами, ожидается один безымянный вектор на точку"
        )
    return vectors.size


def sample_payload(client: QdrantClient, name: str) -> dict:
    """Метаданные первой попавшейся точки: по ним поиск проверяет, чем построена коллекция."""
    points, _ = client.scroll(name, limit=1, with_payload=True, with_vectors=False)
    if not points or points[0].payload is None:
        raise RuntimeError(f"коллекция {name} пуста")
    return points[0].payload


@dataclass(frozen=True)
class CollectionInfo:
    """Что известно о коллекции перед работой."""

    name: str
    size: int
    count: int
    model: str
    revision: str
    input_size: tuple[int, int]


def inspect(client: QdrantClient, name: str) -> CollectionInfo:
    """Проверки коллекции: существует, не пуста, точки помнят модель."""
    if not client.collection_exists(name):
        raise RuntimeError(f"в qdrant нет коллекции {name}")
    count = client.count(name, exact=True).count
    if count == 0:
        raise RuntimeError(f"коллекция {name} пуста")
    payload = sample_payload(client, name)
    missing = [
        f
        for f in (
            FIELD_MODEL,
            FIELD_REVISION,
            FIELD_INPUT_SIZE,
            FIELD_SLUG,
            FIELD_GROUP,
            FIELD_GROUP_SLUGS,
            FIELD_IMAGE,
        )
        if f not in payload
    ]
    if missing:
        raise RuntimeError(
            f"коллекция {name} собрана не build_catalog: в метаданных нет полей {missing}"
        )
    return CollectionInfo(
        name,
        vector_size(client, name),
        count,
        payload[FIELD_MODEL],
        payload[FIELD_REVISION],
        tuple(payload[FIELD_INPUT_SIZE]),
    )


def search(
    client: QdrantClient, name: str, vector: np.ndarray, top_k: int
) -> list[ScoredPoint]:
    """Ближайшие по косинусу точки с метаданными, по убыванию."""
    return client.query_points(
        name, query=vector.tolist(), limit=top_k, with_payload=True, with_vectors=False
    ).points


def retrieve(client: QdrantClient, name: str, slugs: list[str]) -> dict[str, dict]:
    """Метаданные точек по slug; slug, которых в коллекции нет, в ответе отсутствуют."""
    if not slugs:
        return {}
    records = client.retrieve(
        name, ids=[point_id(s) for s in slugs], with_payload=True, with_vectors=False
    )
    return {r.payload[FIELD_SLUG]: r.payload for r in records if r.payload is not None}


def upsert(
    client: QdrantClient, name: str, points: list[PointStruct], batch: int = 256
) -> None:
    """Точки в коллекцию пачками."""
    for start in range(0, len(points), batch):
        client.upsert(name, points=points[start : start + batch], wait=True)
