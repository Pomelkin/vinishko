"""Визуальный поиск: кропы бутылок → кандидаты каталога либо отказ с причиной. Шаг пайплайна после нормализации."""

import json
import re
import time
from datetime import UTC
from datetime import datetime
from pathlib import Path

import numpy as np
from kostyl.utils import setup_logger
from PIL import Image
from qdrant_client import QdrantClient

from vinishko.pred.pipeline.steps.vis_searcher.catalog import FIELD_GROUP
from vinishko.pred.pipeline.steps.vis_searcher.catalog import FIELD_GROUP_SLUGS
from vinishko.pred.pipeline.steps.vis_searcher.catalog import FIELD_IMAGE
from vinishko.pred.pipeline.steps.vis_searcher.catalog import FIELD_SLUG
from vinishko.pred.pipeline.steps.vis_searcher.catalog import CollectionInfo
from vinishko.pred.pipeline.steps.vis_searcher.catalog import collection_slugs
from vinishko.pred.pipeline.steps.vis_searcher.catalog import connect
from vinishko.pred.pipeline.steps.vis_searcher.catalog import fetch_vectors
from vinishko.pred.pipeline.steps.vis_searcher.catalog import inspect
from vinishko.pred.pipeline.steps.vis_searcher.catalog import retrieve
from vinishko.pred.pipeline.steps.vis_searcher.catalog import sample_payload
from vinishko.pred.pipeline.steps.vis_searcher.catalog import search
from vinishko.pred.pipeline.steps.vis_searcher.configs import GroupSearch
from vinishko.pred.pipeline.steps.vis_searcher.configs import TopNSearch
from vinishko.pred.pipeline.steps.vis_searcher.configs import VisSearcherConfig
from vinishko.pred.pipeline.steps.vis_searcher.configs import load_config
from vinishko.pred.pipeline.steps.vis_searcher.device import Device
from vinishko.pred.pipeline.steps.vis_searcher.device import resolve_device
from vinishko.pred.pipeline.steps.vis_searcher.model import Encoder
from vinishko.pred.pipeline.steps.vis_searcher.model import fetch_model
from vinishko.pred.pipeline.steps.vis_searcher.snapshots import make_snapshot_store
from vinishko.pred.pipeline.steps.vis_searcher.snapshots import restore_collection
from vinishko.pred.pipeline.steps.vis_searcher.storage import ImageStore
from vinishko.pred.pipeline.steps.vis_searcher.storage import make_store
from vinishko.pred.pipeline.steps.vis_searcher.storage import read_manifest
from vinishko.pred.pipeline.structs import BottleCandidates
from vinishko.pred.pipeline.structs import BottleCrop
from vinishko.pred.pipeline.structs import Candidate
from vinishko.pred.pipeline.structs import Reason
from vinishko.pred.pipeline.structs import Rejection
from vinishko.pred.pipeline.structs import UnmatchedBottle


logger = setup_logger(fmt="detailed")

Hit = tuple[float, dict, dict[str, float]]
"""Скор позиции — среднее косинусов по входам энкодера, метаданные точки, косинус по каждому входу."""


class SearchReason(Reason):
    """Почему визуальный поиск не дал кандидатов."""

    __stage__ = "search"
    NO_MATCH = (
        "no_match",
        "Нет в каталоге",
        "Похожих позиций в каталоге не нашлось: косинус лучшей ниже порога search.cosine_threshold либо search.group_threshold",
    )


class VisSearcher:
    """Кропы бутылок → ответ на каждый кроп, в том же порядке: BottleCandidates с кандидатами каталога либо UnmatchedBottle с отказом и причиной.

    Входы энкодера — encoder_input конфига: один кроп либо гибрид crop+box_crop, тогда каждый кроп кодируется отдельно, поиск идёт
    в пространстве каждого входа, выдачи объединяются по slug, скор позиции — среднее косинусов, и порог берётся по нему.
    Если коллекции в qdrant нет, она восстанавливается из хранилища snapshots конфига (снапшот кладёт туда dump_collection); нет и снапшота —
    ошибка. При создании сначала дешёвые проверки: контракт модели с Hugging Face, коллекция существует и не пуста, построена той же моделью
    и ревизией, размерность векторов, размер входа и набор входов совпадают. Потом скачивается граф и поднимается энкодер, последним
    хранилище картинок: его manifest.json с тем же отпечатком нормализации, что у коллекции, локальный кэш картинок S3 от этой же
    сборки (иначе сбрасывается), и картинка первой точки читается.
    Нормализатор запроса сверяет с манифестом оркестратор через check_normalization. Без cfg — config.yaml рядом с модулем,
    без device — VIS_SEARCHER_DEV либо автоматически. client — готовый клиент qdrant вместо подключения по конфигу, для тестов.
    """

    def __init__(
        self,
        cfg: VisSearcherConfig | None = None,
        device: Device | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self.cfg = cfg = cfg or load_config()
        self.inputs = list(cfg.encoder_input)
        self.measure = "средний косинус" if len(self.inputs) > 1 else "косинус"
        self.device = device or resolve_device(cfg.device)
        self.files = fetch_model(cfg.model, self.device.precision, cfg.revision)
        self.client = client or connect(cfg.qdrant)
        self.collection = cfg.qdrant.collection
        if not self.client.collection_exists(self.collection):
            self._restore()
        self.info = self._check_collection(inspect(self.client, self.collection))
        self.using = {inp: (inp if self.info.named else None) for inp in self.inputs}
        self.encoder = Encoder(self.files, self.device, cfg.batch_size, cfg.cache_dir)
        self.store: ImageStore = make_store(cfg.images, cfg.cache_dir)
        self.manifest = self._check_store()
        logger.info(
            f"поиск готов: {self.encoder.description}; коллекция {self.info.name}, {self.info.count} точек ×{self.info.size}, "
            f"входы {'+'.join(self.inputs)}; картинки: {self.store.description}; режим {cfg.search.mode}"
        )

    def _restore(self) -> None:
        snapshots = self.cfg.snapshots
        if snapshots is None:
            raise RuntimeError(
                f"в qdrant нет коллекции {self.collection}, а snapshots в конфиге не задан: восстановить её неоткуда"
            )
        restore_collection(
            self.client,
            self.cfg.qdrant,
            self.collection,
            make_snapshot_store(snapshots, self.cfg.cache_dir),
            snapshots.timeout,
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
        if sorted(info.encoder_input) != sorted(self.inputs):
            raise RuntimeError(
                f"коллекция {info.name} собрана по {'+'.join(info.encoder_input)}, а в конфиге encoder_input={'+'.join(self.inputs)}: "
                "пересоберите коллекцию либо верните значение"
            )
        return info

    def _check_store(self) -> dict:
        manifest = read_manifest(self.store)
        if manifest["fingerprint"] != self.info.normalization:
            raise RuntimeError(
                f"картинки в хранилище ({self.store.description}) писала сборка коллекции {manifest['collection']} с нормализацией {manifest['fingerprint']}, "
                f"а коллекция {self.collection} собрана с {self.info.normalization}: пересоберите {self.collection} либо дайте ей своё хранилище"
            )
        self.store.sync_cache(
            manifest
        )  # локальные копии картинок другой сборки под теми же именами — сброс до первого чтения
        name = sample_payload(self.client, self.collection)[FIELD_IMAGE]
        if not self.store.exists(name):
            raise RuntimeError(
                f"в хранилище картинок ({self.store.description}) нет {name} из коллекции {self.collection}"
            )
        self.store.get(name)
        return manifest

    def slugs(self) -> set[str]:
        """Все позиции коллекции: оркестратор сверяет их с каталогом при старте."""
        return collection_slugs(self.client, self.collection)

    def check_normalization(self, normalization: dict) -> None:
        """Нормализатор запроса вырезает кропы так же, как при сборке коллекции: normalization — Normalizer.crop_config; иначе ошибка с перечнем расхождений."""
        changes = config_diff(self.manifest["normalization"], normalization)
        if changes:
            raise RuntimeError(
                f"нормализация не та, что собирала коллекцию {self.collection}: "
                + "; ".join(changes)
                + ". Пересоберите коллекцию либо верните конфиг нормализации"
            )

    def __call__(
        self, crops: list[BottleCrop]
    ) -> list[BottleCandidates | UnmatchedBottle]:
        """Ответ по каждому кропу, в порядке кропов: результатов ровно столько, сколько кропов."""
        if not crops:
            return []
        vectors = {
            inp: self.encoder([getattr(crop, inp) for crop in crops])
            for inp in self.inputs
        }
        results = [
            self._search(crop, {inp: vectors[inp][i] for inp in self.inputs})
            for i, crop in enumerate(crops)
        ]
        if self.cfg.debug_path is not None:
            self.dump(
                results,
                self.cfg.debug_path
                / datetime.now(tz=UTC).astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f"),
            )
        return results

    def warmup(self) -> None:
        """Холостой прогон на белом кропе: энкодер по каждому входу и поиск в qdrant с досчётом векторов — первый запрос не платит за разогрев."""
        started = time.perf_counter()
        blank = np.full((*self.files.input_size, 3), 255, np.uint8)
        query: dict[str, np.ndarray] = {
            inp: self.encoder([blank])[0] for inp in self.inputs
        }
        encoded = time.perf_counter()
        self._hits(query)
        logger.info(
            f"поиск прогрет: энкодер {encoded - started:.2f} с, qdrant {time.perf_counter() - encoded:.2f} с"
        )

    def _search(
        self, crop: BottleCrop, query: dict[str, np.ndarray]
    ) -> BottleCandidates | UnmatchedBottle:
        hits = self._hits(query)
        mode = self.cfg.search
        if isinstance(mode, TopNSearch):
            return self._top_n(crop, hits, mode)
        return self._by_groups(crop, hits, mode)

    def _hits(self, query: dict[str, np.ndarray]) -> list[Hit]:
        """top_k позиций по среднему косинусу входов.

        Поиск в пространстве каждого входа даёт свой top_k, выдачи объединяются по slug, и позиции, пришедшей только из одного
        пространства, недостающий косинус досчитывается точно по её вектору из коллекции. Запрашивать больше top_k не нужно:
        на тесте 2026-09-27 объединение обычных top-10 совпало со слиянием по всему каталогу. С одним входом это обычный поиск.
        """
        scores: dict[str, dict[str, float]] = {}
        payloads: dict[str, dict] = {}
        for inp in self.inputs:
            for point in search(
                self.client,
                self.collection,
                query[inp],
                self.cfg.top_k,
                self.using[inp],
            ):
                if point.payload is None:
                    continue
                slug = point.payload[FIELD_SLUG]
                payloads.setdefault(slug, point.payload)
                scores.setdefault(slug, {})[inp] = float(point.score)
        for inp in self.inputs:
            missing = [slug for slug, known in scores.items() if inp not in known]
            vectors = fetch_vectors(
                self.client, self.collection, missing, self.using[inp]
            )
            for slug in missing:
                if slug not in vectors:
                    raise RuntimeError(
                        f"у точки {slug} коллекции {self.collection} нет вектора {inp}"
                    )
                scores[slug][inp] = float(vectors[slug] @ query[inp])
        hits: list[Hit] = [
            (sum(known.values()) / len(self.inputs), payloads[slug], known)
            for slug, known in scores.items()
        ]
        hits.sort(key=lambda hit: -hit[0])
        return hits[: self.cfg.top_k]

    def _rejected(
        self, crop: BottleCrop, best: float | None, threshold: float, what: str
    ) -> UnmatchedBottle:
        detail = (
            f"{self.measure} лучшей {what} {best:.3f}, порог {threshold:g}"
            if best is not None
            else "выдача коллекции пуста"
        )
        return UnmatchedBottle(crop, Rejection(SearchReason.NO_MATCH, detail))

    def _top_n(
        self, crop: BottleCrop, hits: list[Hit], mode: TopNSearch
    ) -> BottleCandidates | UnmatchedBottle:
        if not hits or hits[0][0] < mode.cosine_threshold:
            return self._rejected(
                crop, hits[0][0] if hits else None, mode.cosine_threshold, "позиции"
            )
        return BottleCandidates(
            crop,
            [
                self._candidate(crop, payload, score, scores, retrieved=True)
                for score, payload, scores in hits
            ],
        )

    def _by_groups(
        self, crop: BottleCrop, hits: list[Hit], mode: GroupSearch
    ) -> BottleCandidates | UnmatchedBottle:
        """Группы по лучшему скору своих позиций; кандидаты — все позиции прошедших групп: пришедшие со своим скором, остальные со скором группы.

        Группа проходит, если она среди top_groups лучших, её скор не ниже group_threshold и, при group_margin, не дальше него от лучшей.
        """
        group_score: dict[str, float] = {}
        members: dict[str, list[str]] = {}
        by_slug: dict[str, Hit] = {}
        for (
            score,
            payload,
            scores,
        ) in (
            hits
        ):  # выдача уже по убыванию, первое вхождение группы — её лучшая позиция
            group = payload[FIELD_GROUP]
            group_score.setdefault(group, score)
            members.setdefault(group, list(payload[FIELD_GROUP_SLUGS]))
            by_slug.setdefault(payload[FIELD_SLUG], (score, payload, scores))
        ranked = sorted(group_score, key=lambda g: -group_score[g])[: mode.top_groups]
        chosen = [g for g in ranked if group_score[g] >= mode.group_threshold]
        if not chosen:
            return self._rejected(
                crop,
                group_score[ranked[0]] if ranked else None,
                mode.group_threshold,
                "группы",
            )
        if mode.group_margin is not None:
            floor = group_score[chosen[0]] - mode.group_margin
            chosen = [g for g in chosen if group_score[g] >= floor]
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
                    crop,
                    by_slug[slug][1],
                    by_slug[slug][0],
                    by_slug[slug][2],
                    retrieved=True,
                )
                for slug in present
            ]
            out += [
                self._candidate(
                    crop, payloads[slug], group_score[group], {}, retrieved=False
                )
                for slug in members[group]
                if slug in payloads
            ]
        return BottleCandidates(crop, out)

    def _candidate(
        self,
        crop: BottleCrop,
        payload: dict,
        score: float,
        scores: dict[str, float],
        retrieved: bool,
    ) -> Candidate:
        return Candidate(
            slug=payload[FIELD_SLUG],
            score=score,
            image=self.store.get(payload[FIELD_IMAGE]),
            crop=crop,
            group=payload[FIELD_GROUP],
            retrieved=retrieved,
            payload=payload,
            scores=scores,
        )

    def dump(
        self, results: list[BottleCandidates | UnmatchedBottle], out: Path
    ) -> None:
        """Разбор глазами в директорию out: оба кропа каждого запроса (поиска и вся бутылка _box), картинки его кандидатов со скором, а в режиме групп и с группой, в имени файла, и results.json с косинусами по входам и причинами отказов."""
        out.mkdir(parents=True, exist_ok=True)
        grouped = isinstance(self.cfg.search, GroupSearch)
        report = []
        for n, result in enumerate(results, 1):
            crop = result.crop
            candidates = (
                result.candidates if isinstance(result, BottleCandidates) else []
            )
            Image.fromarray(crop.crop).save(
                out / f"q{n}_query_{crop.uuid[:8]}.jpg", quality=95
            )
            Image.fromarray(crop.box_crop).save(
                out / f"q{n}_query_{crop.uuid[:8]}_box.jpg", quality=95
            )
            for rank, c in enumerate(candidates, 1):
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
                    "rejected": {
                        "stage": result.rejection.stage,
                        "reason": result.rejection.reason,
                        "detail": result.rejection.detail,
                    }
                    if isinstance(result, UnmatchedBottle)
                    else None,
                    "candidates": [
                        {
                            "slug": c.slug,
                            "group": c.group,
                            "score": c.score,
                            "scores": c.scores,
                            "retrieved": c.retrieved,
                        }
                        for c in candidates
                    ],
                }
            )
        (out / "results.json").write_text(
            json.dumps(
                {
                    "mode": self.cfg.search.mode,
                    "encoder_input": self.inputs,
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


def config_diff(built: dict, now: dict) -> list[str]:
    """Расхождения двух конфигов вида {секция: {ключ: значение}}, строками «секция.ключ: при сборке X, сейчас Y»."""
    flat_built = {
        f"{section}.{key}": value
        for section, keys in built.items()
        for key, value in keys.items()
    }
    flat_now = {
        f"{section}.{key}": value
        for section, keys in now.items()
        for key, value in keys.items()
    }
    changes = []
    for key in sorted(flat_built | flat_now):
        if key not in flat_now:
            changes.append(f"{key}: при сборке {flat_built[key]!r}, сейчас нет")
        elif key not in flat_built:
            changes.append(f"{key}: при сборке нет, сейчас {flat_now[key]!r}")
        elif flat_built[key] != flat_now[key]:
            changes.append(
                f"{key}: при сборке {flat_built[key]!r}, сейчас {flat_now[key]!r}"
            )
    return changes
