from dataclasses import dataclass
from dataclasses import field

import cv2
import matplotlib as mpl

mpl.use("Agg")  # обучение идёт на сервере без дисплея
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from torchmetrics.functional.classification import binary_auroc

KS = (1, 3, 5)
TOP = max(KS)
FPRS = (0.01, 0.05)
CHUNK_BYTES = 1 << 30
"""Бюджет памяти под матрицу косинусов одного куска запросов: кусок тем меньше, чем больше галерея, и полная матрица n×n не собирается никогда."""
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
"""Первые слоты категориальной палитры отчётов scripts/bench_*: порядок фиксирован, цвет привязан к группе запросов."""
PANEL_SIDE = 160
"""Сторона картинки в сетке примеров: полный вход в TensorBoard не нужен, а десятки картинок по 512 px раздувают лог."""


@dataclass
class Misses:
    """Промахи wine/oneshot_noisy: запросы, чей top-1 — не фото их класса. Строки val и noise — в координатах входов evaluate_retrieval."""

    total: int
    """Сколько всего было запросов протокола."""
    queries: torch.Tensor
    """Строка val каждого промахнувшегося запроса."""
    expected: torch.Tensor
    """Строка val фото его класса, лежавшего в галерее."""
    found: torch.Tensor
    """[n, TOP] строки найденного: val для вина, noise для не-вина, см. found_is_noise."""
    found_is_noise: torch.Tensor
    scores: torch.Tensor
    """[n, TOP] косинусы найденного."""
    hits: torch.Tensor
    """[n, TOP] где среди найденного фото класса запроса: первый столбец всегда False."""


@dataclass
class Panel:
    """Одна картинка сетки примеров с подписью и цветом рамки."""

    image: np.ndarray
    caption: str
    color: str | None = None


@dataclass
class RetrievalReport:
    """Итог одной валидации: скаляры для логгера, распределения косинуса top-1 по группам запросов и промахи главного протокола."""

    scalars: dict[str, float] = field(default_factory=dict)
    top1: dict[str, torch.Tensor] = field(default_factory=dict)
    """Косинус top-1 каждого запроса группы: у хорошей модели верные запросы val справа, всё остальное слева."""
    misses: Misses | None = None


@torch.no_grad()
def search(
    queries: torch.Tensor, gallery: torch.Tensor, exclude: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """TOP ближайших по косинусу строк галереи для каждого запроса; векторы L2-нормированы.

    exclude — для каждого запроса номер строки галереи, которую нельзя находить, либо −1: так из выдачи убирается сам запрос.
    """
    scores, ids = [], []
    chunk = max(
        1,
        min(
            len(queries),
            CHUNK_BYTES // max(1, gallery.shape[0] * queries.element_size()),
        ),
    )
    for start in range(0, len(queries), chunk):
        sim = queries[start : start + chunk] @ gallery.T
        if exclude is not None:
            rows = exclude[start : start + chunk]
            own = rows >= 0
            sim[torch.nonzero(own).squeeze(1), rows[own]] = -torch.inf
        top = sim.topk(min(TOP, gallery.shape[0]), dim=1)
        scores.append(top.values)
        ids.append(top.indices)
    return torch.cat(scores), torch.cat(ids)


def protocol_rows(
    labels: torch.Tensor, protocol: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Строки val для галереи и строки-запросы, те же протоколы, что в scripts/bench_common.

    loo — в галерее вся val, запросами идут картинки классов, где их две и больше; попаданием считается любое другое фото класса.
    oneshot — в галерее первая картинка каждого класса, запросами все остальные: как в каталоге, где у позиции одно фото.
    """
    everything = torch.arange(len(labels), device=labels.device)
    if protocol == "loo":
        _, inverse, counts = labels.unique(return_inverse=True, return_counts=True)
        return everything, everything[counts[inverse] >= 2]
    order = torch.argsort(labels, stable=True)
    sorted_labels = labels[order]
    first = torch.ones_like(sorted_labels, dtype=torch.bool)
    first[1:] = sorted_labels[1:] != sorted_labels[:-1]
    in_gallery = torch.zeros(len(labels), dtype=torch.bool, device=labels.device)
    in_gallery[order[first]] = True
    return everything[in_gallery], everything[~in_gallery]


def recall(hits: torch.Tensor) -> dict[str, float]:
    """recall@k: доля запросов, у которых среди первых k найденных есть картинка их класса."""
    return {f"recall@{k}": float(hits[:, :k].any(dim=1).float().mean()) for k in KS}


def rejection(accepted: torch.Tensor, rejected: torch.Tensor) -> dict[str, float]:
    """Насколько косинус top-1 отделяет запросы val от запросов без ответа в галерее.

    auroc — ROC AUC; tpr@fpr — доля принятых запросов val при пороге, который пропускает заданную долю запросов без ответа.
    """
    scores = torch.cat([accepted, rejected])
    target = torch.cat([torch.ones_like(accepted), torch.zeros_like(rejected)]).long()
    out = {
        "auroc": float(binary_auroc(scores.sigmoid(), target))
    }  # сигмоида монотонна и ROC не меняет, а вход вне [0, 1] torchmetrics пропустил бы через неё сам, с предупреждением
    for fpr in FPRS:
        out[f"tpr@fpr{round(fpr * 100)}"] = float(
            (accepted > torch.quantile(rejected, 1 - fpr)).float().mean()
        )
    return out


@torch.no_grad()
def evaluate_retrieval(
    val: torch.Tensor,
    labels: torch.Tensor,
    distractors: torch.Tensor,
    negatives: torch.Tensor,
    catalog: torch.Tensor,
    catalog_queries: torch.Tensor,
) -> RetrievalReport:
    """Метрики ретрива по эмбеддингам одной валидации.

    wine/loo и wine/oneshot — вино WineSensed в двух протоколах на чистой галерее. wine/oneshot_noisy — главный замер: одно фото класса
    в галерее плюс шум, половина не-вина Products-10K; по нему же считается отказ: косинус top-1 запросов val против запросов без ответа —
    вин не из галереи, distractors, и второй половины не-вина, products10k. catalog — каталог OFF, одно чистое фото на товар в галерее
    вместе с тем же шумом, запросы — аугментированные виды этих же фото: замена полевым снимкам, которых для каталога нет.
    """
    report = RetrievalReport()
    noise, negative_queries = (
        negatives[: len(negatives) // 2],
        negatives[len(negatives) // 2 :],
    )
    for protocol in ("loo", "oneshot"):
        gallery_rows, rows = protocol_rows(labels, protocol)
        if not len(rows):
            continue
        position = torch.full((len(labels),), -1, device=labels.device)
        position[gallery_rows] = torch.arange(len(gallery_rows), device=labels.device)
        scores, ids = search(val[rows], val[gallery_rows], position[rows])
        hits = labels[gallery_rows][ids] == labels[rows][:, None]
        report.scalars.update(
            {f"wine/{protocol}/{name}": value for name, value in recall(hits).items()}
        )
        if protocol == "loo":
            continue
        gallery = torch.cat([val[gallery_rows], noise])
        gallery_labels = torch.cat(
            [labels[gallery_rows], torch.full((len(noise),), -1, device=labels.device)]
        )
        scores, ids = search(val[rows], gallery)
        hits = gallery_labels[ids] == labels[rows][:, None]
        report.scalars.update(
            {
                f"wine/oneshot_noisy/{name}": value
                for name, value in recall(hits).items()
            }
        )
        report.top1["val_correct"], report.top1["val_wrong"] = (
            scores[hits[:, 0], 0],
            scores[~hits[:, 0], 0],
        )
        report.misses = collect_misses(
            rows, gallery_rows, labels, len(noise), ids, scores, hits
        )
        for name, queries in (
            ("distractors", distractors),
            ("products10k", negative_queries),
        ):
            if len(queries):
                report.top1[name] = search(queries, gallery)[0][:, 0]
                report.scalars.update(
                    {
                        f"reject/{name}/{key}": value
                        for key, value in rejection(
                            scores[:, 0], report.top1[name]
                        ).items()
                    }
                )
    if len(catalog):
        gallery = torch.cat([catalog, noise])
        _, ids = search(catalog_queries, gallery)
        hits = ids == torch.arange(len(catalog_queries), device=ids.device)[:, None]
        report.scalars.update(
            {f"catalog/{name}": value for name, value in recall(hits).items()}
        )
    report.scalars.update(
        {
            f"cos_top1_median/{name}": float(values.median())
            for name, values in report.top1.items()
            if len(values)
        }
    )
    return report


def collect_misses(
    rows: torch.Tensor,
    gallery_rows: torch.Tensor,
    labels: torch.Tensor,
    noise_size: int,
    ids: torch.Tensor,
    scores: torch.Tensor,
    hits: torch.Tensor,
) -> Misses:
    """Промахи протокола oneshot_noisy из его выдачи; галерея там — val[gallery_rows], за ней noise, и у каждого класса в ней одно фото."""
    miss = ~hits[:, 0]
    owner = torch.empty(int(labels.max()) + 1, dtype=torch.long, device=labels.device)
    owner[labels[gallery_rows]] = gallery_rows
    source = torch.cat([gallery_rows, torch.arange(noise_size, device=labels.device)])
    return Misses(
        total=len(rows),
        queries=rows[miss],
        expected=owner[labels[rows][miss]],
        found=source[ids[miss]],
        found_is_noise=ids[miss] >= len(gallery_rows),
        scores=scores[miss],
        hits=hits[miss],
    )


def panels_figure(rows: list[list[Panel]], title: str) -> Figure:
    """Сетка картинок с подписями: строка на пример, столбцы одинаковы у всех строк."""
    n_rows, n_cols = len(rows), max(len(row) for row in rows)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.9 * n_cols, 2.15 * n_rows + 0.5), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    for row, panels in zip(axes, rows, strict=True):
        for ax, panel in zip(row, panels, strict=False):
            ax.imshow(resize_panel(panel.image))
            ax.set_title(panel.caption, fontsize=7.5, pad=3)
            if panel.color is not None:
                ax.add_patch(
                    Rectangle(
                        (0, 0),
                        1,
                        1,
                        transform=ax.transAxes,
                        fill=False,
                        edgecolor=panel.color,
                        linewidth=3,
                    )
                )
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def resize_panel(image: np.ndarray) -> np.ndarray:
    """Картинка не больше PANEL_SIDE по большей стороне."""
    scale = PANEL_SIDE / max(image.shape[:2])
    if scale >= 1:
        return image
    return cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )


def scores_figure(top1: dict[str, torch.Tensor], title: str) -> Figure:
    """Распределения косинуса top-1 по группам запросов: строка на группу, общая ось, высота нормирована внутри группы."""
    groups = {
        name: values.float().cpu().numpy()
        for name, values in top1.items()
        if len(values)
    }
    lo = np.floor(min(v.min() for v in groups.values()) * 20) / 20
    hi = np.ceil(max(v.max() for v in groups.values()) * 20) / 20
    fig, axes = plt.subplots(
        len(groups), 1, figsize=(8, 1.6 * len(groups) + 0.6), sharex=True, squeeze=False
    )
    for ax, color, (name, values) in zip(
        axes[:, 0], SERIES_COLORS, groups.items(), strict=False
    ):
        ax.hist(values, bins=50, range=(lo, hi), color=color, rwidth=0.88)
        ax.set_title(
            f"{name} · {len(values)} запросов · медиана {np.median(values):.3f}",
            loc="left",
            fontsize=10,
        )
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[-1, 0].set_xlabel("косинус top-1")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return fig
