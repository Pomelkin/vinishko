from collections.abc import Callable
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from open_clip.transform import image_transform
from PIL import Image
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader

from scripts.bench_common.data import Split
from scripts.bench_common.data import Views
from scripts.bench_common.data import init_worker
from scripts.bench_common.parallel import Advance
from scripts.bench_common.parallel import EmbedTask
from scripts.bench_common.parallel import pick_dtype

MODEL = "TULIP-so400m-14-384"
PRETRAINED_TAG = "webli"  # из записи этого тега форк берёт предобработку TULIP: mean и std 0.5, bicubic, squash
RESIZE_MODES = ["squash", "longest", "shortest"]

Preprocess = Callable[[Image.Image], Tensor]


def load_encoder(
    weights: Path,
    device: torch.device,
    dtype: torch.dtype,
    resize_mode: str,
    fill: tuple[int, ...],
) -> tuple[nn.Module, Preprocess, tuple[int, int]]:
    """Визуальная башня TULIP, её предобработка и размер входа.

    create_model_and_transforms форка сначала скачивает с HF базовые веса SigLIP на 3,5 ГБ и лишь потом накладывает чекпоинт TULIP.
    Чекпоинт содержит все ключи модели, поэтому база не нужна: модель строится без весов, визуальная башня грузится строго по ключам.
    Текстовая башня в замере не участвует и на устройство не переносится.
    """
    visual = open_clip.create_model(MODEL, pretrained=None).visual
    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    visual.load_state_dict(
        {
            k.removeprefix("visual."): v
            for k, v in state.items()
            if k.startswith("visual.")
        },
        strict=True,
    )
    cfg = open_clip.get_pretrained_cfg(MODEL, PRETRAINED_TAG)
    size: tuple[int, int] = visual.image_size
    visual = visual.to(device=device, dtype=dtype).eval().requires_grad_(False)
    preprocess = image_transform(
        size,
        is_train=False,
        mean=cfg["mean"],
        std=cfg["std"],
        interpolation=cfg["interpolation"],
        resize_mode=resize_mode,
        fill_color=fill,  # ty: ignore[invalid-argument-type]
    )
    return visual, preprocess, size


@torch.inference_mode()
def embed(
    visual: nn.Module,
    preprocess: Preprocess,
    split: Split,
    render_cfg: dict,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    workers: int,
    advance: Advance,
) -> np.ndarray:
    """L2-нормированные эмбеддинги картинок сплита в float32."""
    loader = DataLoader(
        Views(split, preprocess, render_cfg),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=init_worker,
    )
    batches = []
    for images, rows in loader:  # без перемешивания батчи идут в порядке сплита
        features = F.normalize(
            visual(images.to(device=device, dtype=dtype, non_blocking=True)).float(),
            dim=-1,
        )
        batches.append(features.cpu().numpy())
        advance(len(rows))
    return np.concatenate(batches)


class TulipWorker:
    """Воркер одного устройства: визуальная башня TULIP, по задаче пишет эмбеддинги своей части сплита в npy."""

    def __init__(
        self,
        device: torch.device,
        weights: Path,
        resize_mode: str,
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
        self.visual, self.preprocess, size = load_encoder(
            weights,
            device,
            self.dtype,
            resize_mode,
            tuple(render_cfg["background"]["color"]),
        )
        self.info = {"dtype": str(self.dtype), "вход": f"{size[0]}×{size[1]}"}

    def __call__(self, task: EmbedTask, advance: Advance) -> None:
        vectors = embed(
            self.visual,
            self.preprocess,
            task.split,
            self.render_cfg,
            self.device,
            self.dtype,
            self.batch_size,
            self.workers,
            advance,
        )
        with task.path.open("wb") as f:
            np.save(f, vectors)
