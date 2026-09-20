import numpy as np
import torch
from rich.progress import Progress

from scripts.bench_common.data import Split
from scripts.bench_common.metrics import TOP
from scripts.bench_common.metrics import Retrieval
from scripts.bench_evie.store import TokenStore
from vinishko.pipeline.steps.vis_searcher.utils.maxsim import maxsim_inbatch

QUERY_BATCH = 16
BLOCK_BYTES = 1 << 30  # блок галереи на устройстве
PAIR_BYTES = 1 << 30  # матрица сходств токенов одного вызова MaxSim, если считает эталонный einsum, а не слитое ядро


@torch.inference_mode()
def search(
    gallery: TokenStore,
    queries: TokenStore,
    rows: np.ndarray,
    exclude_self: bool,
    device: torch.device,
    dtype: torch.dtype,
    progress: Progress,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Точный MaxSim каждого запроса со всей галереей: TOP лучших картинок и их скоры.

    Скор — MaxSim пайплайна, делённый на число токенов запроса: средний по токенам запроса максимум косинуса с токенами картинки галереи.
    Порядок выдачи деление не меняет, зато скоры запросов разной длины становятся сравнимы, а без этого порог отказа не поставить.
    Галерея проходит через устройство блоками по BLOCK_BYTES, внутри блока идут батчи запросов; exclude_self убирает из выдачи сам запрос.
    """
    itemsize = torch.empty(0, dtype=dtype).element_size()
    blocks = gallery.blocks(BLOCK_BYTES // (gallery.dim * itemsize))
    task = progress.add_task(name, name=f"поиск {name}", total=len(rows) * len(blocks))
    best_scores = torch.full((len(rows), TOP), -torch.inf, device=device)
    best_ids = torch.full((len(rows), TOP), -1, dtype=torch.long, device=device)
    for block in blocks:
        docs = torch.from_numpy(gallery.padded(block)).to(device=device, dtype=dtype)
        doc_ids = torch.from_numpy(block).to(device)
        for start in range(0, len(rows), QUERY_BATCH):
            batch = rows[start : start + QUERY_BATCH]
            q = torch.from_numpy(queries.padded(batch)).to(device=device, dtype=dtype)
            step = max(1, PAIR_BYTES // (q.shape[0] * q.shape[1] * docs.shape[1] * itemsize))
            scores = torch.cat([maxsim_inbatch(q, docs[j : j + step]).float() for j in range(0, len(docs), step)], dim=1)
            scores /= torch.from_numpy(queries.lengths[batch]).to(device)[:, None]
            if exclude_self:
                scores.masked_fill_(doc_ids[None, :] == torch.from_numpy(batch).to(device)[:, None], -torch.inf)
            merged_scores = torch.cat([best_scores[start : start + len(batch)], scores], dim=1)
            merged_ids = torch.cat([best_ids[start : start + len(batch)], doc_ids.expand(len(batch), -1)], dim=1)
            top = merged_scores.topk(TOP, dim=1)
            best_scores[start : start + len(batch)] = top.values
            best_ids[start : start + len(batch)] = merged_ids.gather(1, top.indices)
            progress.update(task, advance=len(batch))
    return best_scores.cpu().numpy(), best_ids.cpu().numpy()


def retrieve_val(split: Split, store: TokenStore, rows: np.ndarray, device: torch.device, dtype: torch.dtype, progress: Progress) -> Retrieval:
    """Каждый запрос val ищет среди всех остальных картинок val."""
    return Retrieval(split, rows, *search(store, store, rows, True, device, dtype, progress, split.role))


def retrieve_rejects(split: Split, store: TokenStore, gallery: TokenStore, device: torch.device, dtype: torch.dtype, progress: Progress) -> Retrieval:
    """Запросы, которых в галерее нет: вся их выдача ложная, интересен только скор."""
    rows = np.arange(len(split))
    return Retrieval(split, rows, *search(gallery, store, rows, False, device, dtype, progress, split.role))
