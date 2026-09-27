from dataclasses import dataclass
from dataclasses import field

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.bench_common.data import Split
from scripts.bench_common.data import protocol_rows


KS = (1, 3, 5)
TOP = max(KS)
FPRS = (0.01, 0.05)
SIGNALS = {"top1": "{score} top-1", "margin": "отрыв top-1 от top-2"}
"""Сигналы отказа: свойство Retrieval и его название; {score} — имя скора модели, подставляет отчёт."""


@dataclass
class Retrieval:
    """Выдача по запросам одной роли: TOP лучших картинок галереи на запрос, по убыванию скора."""

    split: Split
    rows: np.ndarray
    """Номера картинок-запросов в split."""
    scores: np.ndarray
    ids: np.ndarray
    """Номера найденных картинок в галерее, то есть в сплите val."""

    @property
    def top1(self) -> np.ndarray:
        """Скор top-1."""
        return self.scores[:, 0]

    @property
    def margin(self) -> np.ndarray:
        """Отрыв top-1 от top-2."""
        return self.scores[:, 0] - self.scores[:, 1]


@dataclass
class ModeResult:
    """Замер одного режима входа: целое фото либо кроп нормализации."""

    mode: str
    protocol: str
    gallery: Split
    """Сплит val: номера Retrieval.ids указывают в него при любом протоколе."""
    gallery_size: int
    """Сколько картинок val реально лежит в галерее при этом протоколе."""
    val: Retrieval
    hits: np.ndarray
    """(запросы val, TOP): на этом месте выдачи стоит картинка того же класса, что запрос."""
    total_queries: int
    """Запросов val по этому протоколу до нормализации; знаменатель сквозного recall."""
    rejects: dict[str, Retrieval]
    """Запросы без позитивов в галерее: distractors и negatives."""
    dropped: dict[str, tuple[int, int]]
    """По ролям: сколько картинок нормализация не допустила до энкодера и сколько их было."""
    images: int
    seconds: float
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Идентификатор столбца отчёта."""
        return f"{self.mode}-{self.protocol}"

    @property
    def speed(self) -> float:
        """Картинок в секунду на этапе эмбеддингов, вместе с чтением с диска и предобработкой."""
        return self.images / self.seconds


def hit_matrix(val: Retrieval) -> np.ndarray:
    """Совпадение класса найденной картинки с классом запроса."""
    labels = np.array(val.split.labels)
    return labels[val.ids] == labels[val.rows][:, None]


def rejection_metrics(positives: Retrieval, negatives: Retrieval) -> dict[str, float]:
    """Насколько скор выдачи отделяет запросы val от запросов без ответа в галерее.

    auc — ROC AUC по сигналу; tpr@fpr — доля принятых запросов val при пороге, который пропускает заданную долю запросов без ответа.
    """
    out: dict[str, float] = {}
    for signal in SIGNALS:
        pos, neg = getattr(positives, signal), getattr(negatives, signal)
        out[f"auc_{signal}"] = float(
            roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
        )
        for fpr in FPRS:
            out[f"tpr_{signal}@{fpr}"] = float((pos > np.quantile(neg, 1 - fpr)).mean())
    return out


def make_result(
    mode: str,
    protocol: str,
    raw: dict[str, Split],
    splits: dict[str, Split],
    dropped: dict[str, tuple[int, int]],
    gallery_size: int,
    val: Retrieval,
    rejects: dict[str, Retrieval],
    seconds: float,
) -> ModeResult:
    """Собирает замер режима из выдач и считает метрики. raw — сплиты до нормализации: по ним берётся знаменатель сквозного recall."""
    result = ModeResult(
        mode=mode,
        protocol=protocol,
        gallery=splits["val"],
        gallery_size=gallery_size,
        val=val,
        hits=hit_matrix(val),
        total_queries=len(protocol_rows(raw["val"], protocol)[1]),
        rejects=rejects,
        dropped=dropped,
        images=sum(len(split) for split in splits.values()),
        seconds=seconds,
    )
    summarize(result)
    return result


def summarize(result: ModeResult) -> None:
    """Заполняет result.metrics: recall@k по допущенным запросам, сквозной recall@k и метрики отказа по каждой роли без ответа."""
    for k in KS:
        found = result.hits[:, :k].any(axis=1)
        result.metrics[f"recall@{k}"] = float(found.mean())
        result.metrics[f"e2e_recall@{k}"] = float(found.sum() / result.total_queries)
    for role, rejects in result.rejects.items():
        for name, value in rejection_metrics(result.val, rejects).items():
            result.metrics[f"{role}/{name}"] = value
