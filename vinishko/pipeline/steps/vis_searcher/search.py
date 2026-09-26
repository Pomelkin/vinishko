"""Визуальный поиск: кропы бутылок → кандидаты каталога либо отказ с причиной. Шаг пайплайна после нормализации."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from kostyl.utils import setup_logger
from PIL import Image
from qdrant_client import QdrantClient

from vinishko.pipeline.steps.vis_searcher.catalog import (
    FIELD_GROUP,
    FIELD_GROUP_SLUGS,
    FIELD_IMAGE,
    FIELD_SLUG,
    CollectionInfo,
    connect,
    inspect,
    retrieve,
    sample_payload,
    search,
)
from vinishko.pipeline.steps.vis_searcher.configs import (
    GroupSearch,
    TopNSearch,
    VisSearcherConfig,
    load_config,
)
from vinishko.pipeline.steps.vis_searcher.device import Device, resolve_device
from vinishko.pipeline.steps.vis_searcher.model import Encoder, fetch_model
from vinishko.pipeline.steps.vis_searcher.storage import ImageStore, make_store
from vinishko.pipeline.structs import BottleCrop, Candidate, Reason, SearchResult

logger = setup_logger(fmt="detailed")

Hit = tuple[float, dict]
"""Косинус и метаданные точки из выдачи."""


class SearchReason(Reason):
    """Почему визуальный поиск не дал кандидатов."""

    NO_MATCH = (
        "no_match",
        "Нет в каталоге",
        "Похожих позиций в каталоге не нашлось: косинус лучшей ниже порога search.cosine_threshold либо search.group_threshold",
    )


class VisSearcher:
    """Кропы бутылок → SearchResult на каждый кроп, в том же порядке: кандидаты каталога либо отказ с причиной.

    При создании сначала дешёвые проверки: контракт модели с Hugging Face, коллекция существует и не пуста, построена той же моделью
    и ревизией, размерность векторов и размер входа совпадают. Потом скачивается граф и поднимается энкодер, последней читается картинка
    первой точки из хранилища. Без cfg — config.yaml рядом с модулем, без device — VIS_SEARCHER_DEV либо автоматически.
    client — готовый клиент qdrant вместо подключения по конфигу, для тестов.
    """

    def __init__(
        self,
        cfg: VisSearcherConfig | None = None,
        device: Device | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self.cfg = cfg = cfg or load_config()
        self.device = device or resolve_device()
        self.files = fetch_model(cfg.model, self.device.precision, cfg.revision)
        self.client = client or connect(cfg.qdrant)
        self.collection = cfg.qdrant.collection
        self.info = self._check_collection(inspect(self.client, self.collection))
        self.encoder = Encoder(self.files, self.device, cfg.batch_size, cfg.cache_dir)
        self.store: ImageStore = make_store(cfg.images, cfg.cache_dir)
        self._check_store()
        logger.info(
            f"поиск готов: {self.encoder.description}; коллекция {self.info.name}, {self.info.count} векторов ×{self.info.size}; "
            f"картинки: {self.store.description}; режим {cfg.search.mode}"
        )

    def _check_collection(self, info: CollectionInfo) -> CollectionInfo:
        files = self.files
        if info.model != files.repo or info.revision != files.revision:
            raise RuntimeError(
                f"коллекция {info.name} построена моделью {info.model}@{info.revision[:8]}, а загружена {files.repo}@{files.revision[:8]}: "
                "пересоберите коллекцию либо задайте model и revision этой коллекции в конфиге"
            )
        if info.size != files.embed_dim:
            raise RuntimeError(
                f"в коллекции {info.name} векторы ×{info.size}, у модели embed_dim={files.embed_dim}"
            )
        if info.input_size != files.input_size:
            raise RuntimeError(
                f"коллекция {info.name} построена со входом {info.input_size}, у модели {files.input_size}"
            )
        return info

    def _check_store(self) -> None:
        name = sample_payload(self.client, self.collection)[FIELD_IMAGE]
        if not self.store.exists(name):
            raise RuntimeError(
                f"в хранилище картинок ({self.store.description}) нет {name} из коллекции {self.collection}"
            )
        self.store.get(name)

    def __call__(self, crops: list[BottleCrop]) -> list[SearchResult]:
        """Ответ по каждому кропу, в порядке кропов: результатов ровно столько, сколько кропов."""
        if not crops:
            return []
        vectors = self.encoder([crop.crop for crop in crops])
        results = [
            self._search(crop, vector)
            for crop, vector in zip(crops, vectors, strict=True)
        ]
        if self.cfg.debug_path is not None:
            self._dump(self.cfg.debug_path, results)
        return results

    def _search(self, crop: BottleCrop, vector: np.ndarray) -> SearchResult:
        hits = [
            (float(p.score), p.payload)
            for p in search(self.client, self.collection, vector, self.cfg.top_k)
            if p.payload is not None
        ]
        mode = self.cfg.search
        if isinstance(mode, TopNSearch):
            return self._top_n(crop, hits, mode)
        return self._by_groups(crop, hits, mode)

    def _rejected(
        self, crop: BottleCrop, best: float | None, threshold: float, what: str
    ) -> SearchResult:
        detail = (
            f"косинус лучшей {what} {best:.3f}, порог {threshold:g}"
            if best is not None
            else "выдача коллекции пуста"
        )
        return SearchResult(crop, [], crop.reject(SearchReason.NO_MATCH, detail))

    def _top_n(
        self, crop: BottleCrop, hits: list[Hit], mode: TopNSearch
    ) -> SearchResult:
        if not hits or hits[0][0] < mode.cosine_threshold:
            return self._rejected(
                crop, hits[0][0] if hits else None, mode.cosine_threshold, "позиции"
            )
        return SearchResult(
            crop,
            [
                self._candidate(crop, payload, score, retrieved=True)
                for score, payload in hits
            ],
        )

    def _by_groups(
        self, crop: BottleCrop, hits: list[Hit], mode: GroupSearch
    ) -> SearchResult:
        """Группы по лучшему косинусу своих векторов; кандидаты — все позиции прошедших групп: пришедшие со своим косинусом, остальные с косинусом группы."""
        group_score: dict[str, float] = {}
        members: dict[str, list[str]] = {}
        by_slug: dict[str, Hit] = {}
        for (
            score,
            payload,
        ) in hits:  # выдача уже по убыванию, первое вхождение группы — её лучший вектор
            group = payload[FIELD_GROUP]
            group_score.setdefault(group, score)
            members.setdefault(group, list(payload[FIELD_GROUP_SLUGS]))
            by_slug.setdefault(payload[FIELD_SLUG], (score, payload))
        ranked = sorted(group_score, key=lambda g: -group_score[g])[: mode.top_groups]
        chosen = [g for g in ranked if group_score[g] >= mode.group_threshold]
        if not chosen:
            return self._rejected(
                crop,
                group_score[ranked[0]] if ranked else None,
                mode.group_threshold,
                "группы",
            )
        absent = [slug for g in chosen for slug in members[g] if slug not in by_slug]
        payloads = retrieve(self.client, self.collection, absent)
        out: list[Candidate] = []
        for group in chosen:
            present = sorted(
                (slug for slug in members[group] if slug in by_slug),
                key=lambda slug: -by_slug[slug][0],
            )
            out += [
                self._candidate(
                    crop, by_slug[slug][1], by_slug[slug][0], retrieved=True
                )
                for slug in present
            ]
            out += [
                self._candidate(
                    crop, payloads[slug], group_score[group], retrieved=False
                )
                for slug in members[group]
                if slug in payloads
            ]
        return SearchResult(crop, out)

    def _candidate(
        self, crop: BottleCrop, payload: dict, score: float, retrieved: bool
    ) -> Candidate:
        return Candidate(
            slug=payload[FIELD_SLUG],
            score=score,
            image=self.store.get(payload[FIELD_IMAGE]),
            crop=crop,
            group=payload[FIELD_GROUP],
            retrieved=retrieved,
            payload=payload,
        )

    def _dump(self, root: Path, results: list[SearchResult]) -> None:
        """Директория с меткой времени: кроп каждого запроса и картинки его кандидатов с косинусом, а в режиме групп и с группой, в имени файла."""
        out = root / datetime.now(tz=UTC).astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f")
        out.mkdir(parents=True, exist_ok=True)
        grouped = isinstance(self.cfg.search, GroupSearch)
        report = []
        for n, result in enumerate(results, 1):
            crop = result.crop
            Image.fromarray(crop.crop).save(
                out / f"q{n}_query_{crop.uuid[:8]}.jpg", quality=95
            )
            for rank, c in enumerate(result.candidates, 1):
                parts = [
                    f"q{n}",
                    f"{rank:02d}",
                    *([safe(c.group)] if grouped else []),
                    safe(c.slug),
                    f"{c.score:.3f}",
                    *([] if c.retrieved else ["bygroup"]),
                ]
                Image.fromarray(c.image).save(
                    out / ("_".join(parts) + ".jpg"), quality=95
                )
            report.append(
                {
                    "query": crop.uuid,
                    "bottle_score": crop.score,
                    "rejected": None
                    if result.rejected is None
                    else {
                        "reason": result.rejected.reason,
                        "detail": result.rejected.detail,
                    },
                    "candidates": [
                        {
                            "slug": c.slug,
                            "group": c.group,
                            "score": c.score,
                            "retrieved": c.retrieved,
                        }
                        for c in result.candidates
                    ],
                }
            )
        (out / "results.json").write_text(
            json.dumps(
                {
                    "mode": self.cfg.search.mode,
                    "top_k": self.cfg.top_k,
                    "queries": report,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        logger.info(f"разбор поиска записан в {out}")


def safe(name: str) -> str:
    """Строка, пригодная для имени файла."""
    return re.sub(r"[^\w.-]+", "_", name)[:80]
