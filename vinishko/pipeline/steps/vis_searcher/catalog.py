"""Коллекция векторов каталога в qdrant: схема точки, подключение, создание, поиск и проверки."""

import hashlib
import json
import os
import uuid
from collections.abc import Sequence
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
FIELD_MODEL, FIELD_REVISION, FIELD_INPUT_SIZE, FIELD_ENCODER_INPUT = (
    "model",
    "model_revision",
    "input_size",
    "encoder_input",
)
"""Чем построена коллекция; лежит в каждой точке, поиск сверяет с загруженной моделью. encoder_input — список входов энкодера,
по именованному вектору на каждый; у сборок до 2026-09-27 строка и один безымянный вектор."""
FIELD_NORMALIZATION = "normalization"
"""Отпечаток crop_config нормализатора при сборке; тот же в manifest.json хранилища картинок, поиск сверяет оба с нормализатором запроса."""


def normalization_fingerprint(normalization: dict) -> str:
    """Отпечаток части конфига нормализации, определяющей кропы: sha256 канонического JSON, первые 16 знаков."""
    canonical = json.dumps(
        normalization, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def point_id(slug: str) -> str:
    """Идентификатор точки по slug."""
    return str(uuid.uuid5(POINT_NAMESPACE, slug))


def connect(cfg: QdrantConfig) -> QdrantClient:
    """Клиент qdrant: встроенный по пути либо сервер."""
    if cfg.path is not None:
        return QdrantClient(path=str(cfg.path))
    return QdrantClient(
        host=cfg.host,
        port=cfg.port,
        https=cfg.https,
        api_key=os.environ.get("QDRANT_API_KEY"),
        timeout=int(cfg.timeout),
    )


def create_collection(
    client: QdrantClient, name: str, size: int, inputs: Sequence[str]
) -> None:
    """Новая коллекция: именованный вектор на каждый вход энкодера, у каждого своё пространство и индекс; векторы L2-нормированные, метрика — косинус."""
    client.create_collection(
        name,
        vectors_config={
            inp: VectorParams(size=size, distance=Distance.COSINE) for inp in inputs
        },
    )


def vector_spaces(client: QdrantClient, name: str) -> dict[str | None, int]:
    """Пространства коллекции: имя вектора → размерность. У коллекции с одним безымянным вектором, сборки до 2026-09-27, ключ None."""
    vectors = client.get_collection(name).config.params.vectors
    if vectors is None:
        raise RuntimeError(f"в коллекции {name} нет плотных векторов")
    if isinstance(vectors, VectorParams):
        return {None: vectors.size}
    return {key: params.size for key, params in vectors.items()}


def sample_payload(client: QdrantClient, name: str) -> dict:
    """Метаданные первой попавшейся точки: по ним поиск проверяет, чем построена коллекция."""
    points, _ = client.scroll(name, limit=1, with_payload=True, with_vectors=False)
    if not points or points[0].payload is None:
        raise RuntimeError(f"коллекция {name} пуста")
    return points[0].payload


def encoder_inputs(payload: dict) -> list[str]:
    """Входы энкодера из метаданных точки: список; у сборок до 2026-09-27 строка, а без поля — crop."""
    raw = payload.get(FIELD_ENCODER_INPUT, "crop")
    return [raw] if isinstance(raw, str) else list(raw)


@dataclass(frozen=True, slots=True)
class CollectionInfo:
    """Что известно о коллекции перед работой."""

    name: str
    size: int
    count: int
    model: str
    revision: str
    input_size: tuple[int, int]
    encoder_input: list[str]
    """Входы энкодера при сборке, по пространству векторов на каждый: crop, box_crop либо оба."""
    named: bool
    """Векторы именованные, по входу; False у коллекции с одним безымянным вектором, сборки до 2026-09-27."""
    normalization: str
    """Отпечаток crop_config нормализатора при сборке."""


def inspect(client: QdrantClient, name: str) -> CollectionInfo:
    """Проверки коллекции: существует, не пуста, точки помнят модель и нормализацию, пространства векторов соответствуют входам."""
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
    if FIELD_NORMALIZATION not in payload:
        raise RuntimeError(
            f"коллекция {name} собрана до появления manifest.json (2026-09-27): чем нормализованы её кропы, неизвестно; пересоберите её build_catalog"
        )
    spaces = vector_spaces(client, name)
    sizes = set(spaces.values())
    if len(sizes) != 1:
        raise RuntimeError(
            f"в коллекции {name} векторы разной размерности {spaces}, ожидается одна модель на все входы"
        )
    inputs = encoder_inputs(payload)
    names = [key for key in spaces if key is not None]
    named = len(names) == len(spaces)
    if named and sorted(names) != sorted(inputs):
        raise RuntimeError(
            f"в коллекции {name} пространства векторов {sorted(names)}, а метаданные говорят encoder_input={inputs}"
        )
    if not named and len(inputs) != 1:
        raise RuntimeError(
            f"в коллекции {name} один безымянный вектор, а метаданные говорят encoder_input={inputs}"
        )
    return CollectionInfo(
        name,
        sizes.pop(),
        count,
        payload[FIELD_MODEL],
        payload[FIELD_REVISION],
        tuple(payload[FIELD_INPUT_SIZE]),
        inputs,
        named,
        payload[FIELD_NORMALIZATION],
    )


def search(
    client: QdrantClient,
    name: str,
    vector: np.ndarray,
    top_k: int,
    using: str | None = None,
) -> list[ScoredPoint]:
    """Ближайшие по косинусу точки пространства using с метаданными, по убыванию; using=None — единственный безымянный вектор."""
    return client.query_points(
        name,
        query=vector.tolist(),
        using=using,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    ).points


def fetch_vectors(
    client: QdrantClient, name: str, slugs: list[str], using: str | None = None
) -> dict[str, np.ndarray]:
    """Векторы пространства using по slug, L2-нормированные; slug, которых в коллекции нет, в ответе отсутствуют."""
    if not slugs:
        return {}
    records = client.retrieve(
        name,
        ids=[point_id(s) for s in slugs],
        with_payload=[FIELD_SLUG],
        with_vectors=[using] if using is not None else True,
    )
    out: dict[str, np.ndarray] = {}
    for r in records:
        if r.payload is None or r.vector is None:
            continue
        raw = r.vector
        if using is not None:
            if not isinstance(raw, dict):
                raise RuntimeError(
                    f"у точки {r.payload[FIELD_SLUG]} коллекции {name} безымянный вектор, ожидалось пространство {using}"
                )
            raw = raw[using]
        vector = np.asarray(raw, dtype=np.float32)
        out[r.payload[FIELD_SLUG]] = vector / max(float(np.linalg.norm(vector)), 1e-12)
    return out


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
