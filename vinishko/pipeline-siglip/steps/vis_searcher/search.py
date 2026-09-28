"""Адаптация VisSearcher к готовой dense-коллекции с payload slug/photo."""

import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, ScoredPoint, VectorParams

from vinishko.pipeline.steps.vis_searcher.catalog import CollectionInfo, sample_payload
from vinishko.pipeline.steps.vis_searcher.configs import LocalImagesConfig, VisSearcherConfig
from vinishko.pipeline.steps.vis_searcher.device import Device, resolve_device
from vinishko.pipeline.steps.vis_searcher.model import fetch_model
from vinishko.pipeline.steps.vis_searcher.search import VisSearcher as BaseSearcher, logger
from vinishko.pipeline.steps.vis_searcher.storage import make_store
from vinishko.pipeline.structs import BottleCandidates, BottleCrop, Candidate, UnmatchedBottle

from .configs import load_config
from .model import Encoder


class VisSearcher(BaseSearcher):
    """Батч поиска, пороги, группы и dump наследуются от обычного pipeline."""

    def __init__(self, cfg: VisSearcherConfig | None = None, device: Device | None = None,
                 client: QdrantClient | None = None) -> None:
        self.cfg = cfg = cfg or load_config()
        self.client = client if client is not None else (
            QdrantClient(path=str(cfg.qdrant.path)) if cfg.qdrant.path is not None else
            QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port, https=cfg.qdrant.https,
                         timeout=int(cfg.qdrant.timeout), trust_env=False)
        )
        self.collection = cfg.qdrant.collection
        try:
            info = self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            if not isinstance(vectors, VectorParams) or vectors.multivector_config is not None:
                raise ValueError("нужен один безымянный dense-вектор на точку")
            if vectors.distance != Distance.COSINE:
                raise ValueError("SigLIP поиск ожидает метрику Cosine")
            payload = sample_payload(self.client, self.collection)
            self.device = device or resolve_device(cfg.device)
            self.files = fetch_model(cfg.model, self.device.precision, cfg.revision)
            if vectors.size != self.files.embed_dim:
                raise ValueError(f"размерность коллекции {vectors.size}, модели {self.files.embed_dim}")
            self._check_model_metadata(payload)
            self.info = CollectionInfo(self.collection, vectors.size, info.points_count or 0,
                                       self.files.repo, self.files.revision, self.files.input_size, cfg.encoder_input)
            images = cfg.images if cfg.images is not None else cfg.reference_images
            if images is None:
                raise ValueError("нужны images или reference_images для картинок кандидатов")
            self.image_field = "image" if cfg.images is not None else "photo"
            self.store = make_store(images, cfg.cache_dir)
            self._validate_payload(payload)
            self._check_store()
            self.encoder = Encoder(self.files, self.device, cfg.batch_size, cfg.cache_dir, cfg.cpu_batch_size)
        except BaseException:
            if client is None:
                self.client.close()
            raise
        logger.info(f"SigLIP поиск готов: {self.encoder.description}; {self.collection}, {self.info.count} векторов")

    def _check_model_metadata(self, payload: dict) -> None:
        """DenseRetriever не записывал модель; присутствующие метаданные проверяем."""
        config = json.loads((self.files.root / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "siglip2_vision":
            raise ValueError("ожидается экспорт siglip2_vision")
        if "model" in payload and payload["model"] not in {self.files.repo, config.get("architecture")}:
            raise ValueError("коллекция построена другой моделью")
        if "model_revision" in payload and payload["model_revision"] != self.files.revision:
            raise ValueError("ревизия модели коллекции не совпадает с экспортом")
        if "input_size" in payload and tuple(payload["input_size"]) != self.files.input_size:
            raise ValueError("размер входа коллекции не совпадает с экспортом")
        if "model" not in payload:
            logger.warning("В коллекции нет метаданных модели: проверены размерность и Cosine; происхождение векторов не подтверждено")

    def _image_name(self, payload: dict) -> str:
        name = super()._image_name(payload)
        images = self.cfg.images if self.cfg.images is not None else self.cfg.reference_images
        if isinstance(images, LocalImagesConfig):
            root = images.dir.resolve()
            if Path(name).is_absolute() or not (root / name).resolve().is_relative_to(root):
                raise ValueError(f"недопустимый путь картинки в payload: {name!r}")
        return name

    def _validate_payload(self, payload: dict) -> None:
        if not isinstance(payload.get("slug"), str) or not payload["slug"].strip():
            raise ValueError("у точки коллекции нет непустого slug")
        self._image_name(payload)

    def _search(self, crop: BottleCrop, points: list[ScoredPoint]) -> BottleCandidates | UnmatchedBottle:
        hits = []
        for point in points:
            payload = dict(point.payload or {})
            self._validate_payload(payload)
            payload.setdefault("group", payload["slug"])
            payload.setdefault("group_slugs", [payload["slug"]])
            hits.append(point.model_copy(update={"payload": payload}))
        return super()._search(crop, hits)

    def _candidate(self, crop: BottleCrop, payload: dict, score: float, retrieved: bool) -> Candidate:
        self._validate_payload(payload)
        return super()._candidate(crop, payload, score, retrieved)

    def close(self) -> None:
        self.client.close()
