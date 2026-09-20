import faiss
import numpy as np
from rich.progress import Progress

from scripts.bench_common.data import Split
from scripts.bench_common.metrics import TOP
from scripts.bench_common.metrics import Retrieval

SEARCH_BATCH = 2048


def search(gallery: np.ndarray, queries: np.ndarray, k: int, progress: Progress, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Точный поиск по косинусу: векторы L2-нормированы, IndexFlatIP считает скалярное произведение."""
    index = faiss.IndexFlatIP(gallery.shape[1])
    index.add(gallery)
    task = progress.add_task(name, name=f"поиск {name}", total=len(queries))
    scores, ids = [], []
    for start in range(0, len(queries), SEARCH_BATCH):
        batch = queries[start : start + SEARCH_BATCH]
        s, i = index.search(batch, k)
        scores.append(s)
        ids.append(i)
        progress.update(task, advance=len(batch))
    return np.concatenate(scores), np.concatenate(ids)


def drop_self(scores: np.ndarray, ids: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out: убирает из выдачи сам запрос и оставляет TOP следующих. На входе выдача на одну позицию длиннее TOP."""
    order = np.argsort(ids == rows[:, None], axis=1, kind="stable")[:, :TOP]  # запрос уезжает в хвост, порядок остальных сохраняется
    return np.take_along_axis(scores, order, axis=1), np.take_along_axis(ids, order, axis=1)


def retrieve_val(split: Split, vectors: np.ndarray, rows: np.ndarray, progress: Progress) -> Retrieval:
    """Каждый запрос val ищет среди всех остальных картинок val."""
    scores, ids = search(vectors, vectors[rows], TOP + 1, progress, split.role)
    return Retrieval(split, rows, *drop_self(scores, ids, rows))


def retrieve_rejects(split: Split, vectors: np.ndarray, gallery: np.ndarray, progress: Progress) -> Retrieval:
    """Запросы, которых в галерее нет: вся их выдача ложная, интересен только скор."""
    scores, ids = search(gallery, vectors, TOP, progress, split.role)
    return Retrieval(split, np.arange(len(split)), scores, ids)
