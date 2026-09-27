import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from vinishko.pipeline.steps.vis_searcher.modeling_colqwen3_5 import DEFAULT_HEAD_DIMS
from vinishko.pipeline.steps.vis_searcher.modeling_colqwen3_5 import set_active_head

from scripts.bench_common.data import Split
from scripts.bench_common.data import Views
from scripts.bench_common.data import init_worker
from scripts.bench_common.parallel import Advance
from scripts.bench_common.parallel import EmbedTask
from scripts.bench_common.parallel import pick_dtype
from scripts.bench_evie.search import SearchTask
from scripts.bench_evie.search import search
from vinishko.pipeline.steps.vis_searcher import ColQwen3_5
from vinishko.pipeline.steps.vis_searcher import ColQwen3_5Processor


os.environ.setdefault(
    "TOKENIZERS_PARALLELISM", "false"
)  # токенизатор работает в воркерах даталоадера: свои потоки после fork ему ни к чему

MODEL_ID = "tencent/EVIE-4.5B"
HEAD_DIMS = DEFAULT_HEAD_DIMS
ATTN_IMPLS = ["flash_attention_2", "sdpa", "eager"]


def pick_attn(device: torch.device, dtype: torch.dtype) -> str:
    """FlashAttention 2 там, где он работает: на CUDA в bfloat16; иначе SDPA."""
    return (
        "flash_attention_2"
        if device.type == "cuda" and dtype is torch.bfloat16
        else "sdpa"
    )


def load_model(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    attn: str,
    dim: int,
    max_visual_tokens: int | None,
) -> tuple[ColQwen3_5, ColQwen3_5Processor]:
    """EVIE с двунаправленным вниманием и выбранной шириной Prefix-MRL, как в README модели, и её процессор."""
    model = (
        ColQwen3_5.from_pretrained(model_id, dtype=dtype, attn_implementation=attn)
        .to(device)
        .eval()
    )  # без device_map: он тянет за собой accelerate
    model.enable_bidirectional_attention()
    set_active_head(model, dim)
    if max_visual_tokens is None:
        return model, ColQwen3_5Processor.from_pretrained(model_id)
    return model, ColQwen3_5Processor.from_pretrained(
        model_id, max_num_visual_tokens=max_visual_tokens
    )


def keep_image(img: Image.Image) -> Image.Image:
    """Предобработку делает процессор сразу на батч, в collate."""
    return img


class Collate:
    """Батч картинок → входы модели. Работает в воркере даталоадера: ресайз и токенизация не должны занимать главный процесс."""

    def __init__(self, processor: ColQwen3_5Processor) -> None:
        self.processor = processor

    def __call__(
        self, items: list[tuple[Image.Image, int]]
    ) -> tuple[dict[str, torch.Tensor], list[int]]:
        return dict(self.processor.process_images([img for img, _ in items])), [
            row for _, row in items
        ]


@torch.inference_mode()
def embed(
    model: ColQwen3_5,
    processor: ColQwen3_5Processor,
    split: Split,
    render_cfg: dict,
    path: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    advance: Advance,
) -> np.ndarray:
    """Пишет в path токены всех картинок сплита подряд, в float16, и возвращает число токенов каждой картинки.

    Токены картинки — все позиции её последовательности, включая служебные и промпт процессора: так же документы кодирует пайплайн.
    Векторы L2-нормированы моделью.
    """
    loader = DataLoader(
        Views(split, keep_image, render_cfg),
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=Collate(processor),
        pin_memory=device.type == "cuda",
        worker_init_fn=init_worker,
    )
    lengths: list[int] = []
    with path.open("wb") as f:
        for batch, rows in loader:
            if rows != list(range(len(lengths), len(lengths) + len(rows))):
                raise RuntimeError(
                    "даталоадер отдал картинки не по порядку: файл токенов пишется подряд"
                )
            inputs: dict[str, Any] = {
                k: v.to(device, non_blocking=True) for k, v in batch.items()
            }
            model.rope_deltas = (
                None  # иначе позиции следующего батча считаются со сдвигом предыдущего
            )
            tokens = model(**inputs)
            for vectors, mask in zip(
                tokens, inputs["attention_mask"].bool(), strict=True
            ):
                kept = vectors[mask].to(torch.float16).cpu().numpy()
                f.write(kept.tobytes())
                lengths.append(len(kept))
            advance(len(rows))
    return np.array(lengths, np.int64)


@dataclass(frozen=True)
class UnloadTask:
    """Выгрузить модель: эмбеддинги посчитаны, память устройства нужна поиску."""


class EvieWorker:
    """Воркер одного устройства: сначала считает эмбеддинги своей части сплитов, затем выгружает модель и ищет MaxSim по своей доле запросов."""

    def __init__(
        self,
        device: torch.device,
        model_id: str,
        attn: str | None,
        dim: int,
        max_visual_tokens: int | None,
        render_cfg: dict,
        batch_size: int,
        workers: int,
    ) -> None:
        self.device, self.render_cfg, self.batch_size, self.workers = (
            device,
            render_cfg,
            batch_size,
            workers,
        )
        self.dtype = pick_dtype(device)
        attn = attn or pick_attn(device, self.dtype)
        self.model: ColQwen3_5 | None
        self.model, self.processor = load_model(
            model_id, device, self.dtype, attn, dim, max_visual_tokens
        )
        self.info = {"dtype": str(self.dtype), "attn": attn}

    def __call__(
        self, task: EmbedTask | UnloadTask | SearchTask, advance: Advance
    ) -> Any:
        if isinstance(task, SearchTask):
            return search(
                task.gallery,
                task.gallery_rows,
                task.queries,
                task.rows,
                task.exclude_self,
                self.device,
                self.dtype,
                advance,
            )
        if isinstance(task, UnloadTask):
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            return None
        if self.model is None:
            raise RuntimeError("модель уже выгружена: эмбеддинги считаются до поиска")
        return embed(
            self.model,
            self.processor,
            task.split,
            self.render_cfg,
            task.path,
            self.device,
            self.batch_size,
            self.workers,
            advance,
        )
