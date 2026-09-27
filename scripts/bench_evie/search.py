from dataclasses import dataclass

import numpy as np
import torch
from vinishko.pipeline.steps.vis_searcher.utils.maxsim import maxsim_inbatch

from scripts.bench_common.metrics import TOP
from scripts.bench_common.parallel import Advance
from scripts.bench_evie.store import TokenStore


QUERY_BATCH = 16
BLOCK_BYTES = 1 << 30  # блок галереи на устройстве
PAIR_BYTES = (
    1 << 30
)  # матрица сходств токенов одного вызова MaxSim, если считает эталонный einsum, а не слитое ядро


def gallery_blocks(
    gallery: TokenStore, gallery_rows: np.ndarray, dtype: torch.dtype
) -> list[np.ndarray]:
    """Блоки галереи, которые по очереди проходят через устройство; их число нужно и главному процессу, для длины бара."""
    return gallery.blocks(gallery_rows, BLOCK_BYTES // (gallery.dim * dtype.itemsize))


@dataclass(frozen=True)
class SearchTask:
    """Найти для запросов rows хранилища queries лучшие картинки галереи."""

    gallery: TokenStore
    gallery_rows: np.ndarray
    """Какие картинки хранилища val участвуют в поиске: вся val либо одно фото на класс."""
    queries: TokenStore
    rows: np.ndarray
    exclude_self: bool


@torch.inference_mode()
def search(
    gallery: TokenStore,
    gallery_rows: np.ndarray,
    queries: TokenStore,
    rows: np.ndarray,
    exclude_self: bool,
    device: torch.device,
    dtype: torch.dtype,
    advance: Advance,
) -> tuple[np.ndarray, np.ndarray]:
    """Точный MaxSim каждого запроса со всеми картинками gallery_rows галереи: TOP лучших и их скоры, номера — в нумерации хранилища галереи.

    Скор — MaxSim пайплайна, делённый на число токенов запроса: средний по токенам запроса максимум косинуса с токенами картинки галереи.
    Порядок выдачи деление не меняет, зато скоры запросов разной длины становятся сравнимы, а без этого порог отказа не поставить.
    Галерея проходит через устройство блоками по BLOCK_BYTES, внутри блока идут батчи запросов; exclude_self убирает из выдачи сам запрос.
    """
    itemsize = dtype.itemsize
    blocks = gallery_blocks(gallery, gallery_rows, dtype)
    best_scores = torch.full((len(rows), TOP), -torch.inf, device=device)
    best_ids = torch.full((len(rows), TOP), -1, dtype=torch.long, device=device)
    for block in blocks:
        docs = torch.from_numpy(gallery.padded(block)).to(device=device, dtype=dtype)
        doc_ids = torch.from_numpy(block).to(device)
        for start in range(0, len(rows), QUERY_BATCH):
            batch = rows[start : start + QUERY_BATCH]
            q = torch.from_numpy(queries.padded(batch)).to(device=device, dtype=dtype)
            step = max(
                1, PAIR_BYTES // (q.shape[0] * q.shape[1] * docs.shape[1] * itemsize)
            )
            scores = torch.cat(
                [
                    maxsim_inbatch(q, docs[j : j + step]).float()
                    for j in range(0, len(docs), step)
                ],
                dim=1,
            )
            scores /= torch.from_numpy(queries.lengths[batch]).to(device)[:, None]
            if exclude_self:
                scores.masked_fill_(
                    doc_ids[None, :] == torch.from_numpy(batch).to(device)[:, None],
                    -torch.inf,
                )
            merged_scores = torch.cat(
                [best_scores[start : start + len(batch)], scores], dim=1
            )
            merged_ids = torch.cat(
                [best_ids[start : start + len(batch)], doc_ids.expand(len(batch), -1)],
                dim=1,
            )
            top = merged_scores.topk(TOP, dim=1)
            best_scores[start : start + len(batch)] = top.values
            best_ids[start : start + len(batch)] = merged_ids.gather(1, top.indices)
            advance(len(batch))
    return best_scores.cpu().numpy(), best_ids.cpu().numpy()
