import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import open_clip
import rich_click as click
import torch
import torch.nn.functional as F
from open_clip.transform import image_transform
from PIL import Image
from rich.progress import Progress
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader

from scripts.bench_common.data import Split
from scripts.bench_common.data import Views
from scripts.bench_common.data import init_worker

MODEL = "TULIP-so400m-14-384"
PRETRAINED_TAG = "webli"  # из записи этого тега форк берёт предобработку TULIP: mean и std 0.5, bicubic, squash
RESIZE_MODES = ["squash", "longest", "shortest"]

Preprocess = Callable[[Image.Image], Tensor]


def load_encoder(weights: Path, device: torch.device, dtype: torch.dtype, resize_mode: str, fill: tuple[int, ...]) -> tuple[nn.Module, Preprocess, tuple[int, int]]:
    """Визуальная башня TULIP, её предобработка и размер входа.

    create_model_and_transforms форка сначала скачивает с HF базовые веса SigLIP на 3,5 ГБ и лишь потом накладывает чекпоинт TULIP.
    Чекпоинт содержит все ключи модели, поэтому база не нужна: модель строится без весов, визуальная башня грузится строго по ключам.
    Текстовая башня в замере не участвует и на устройство не переносится.
    """
    visual = open_clip.create_model(MODEL, pretrained=None).visual
    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    visual.load_state_dict({k.removeprefix("visual."): v for k, v in state.items() if k.startswith("visual.")}, strict=True)
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
    progress: Progress,
) -> tuple[np.ndarray, float]:
    """L2-нормированные эмбеддинги картинок сплита в float32 и время прогона в секундах вместе с чтением и предобработкой."""
    loader = DataLoader(
        Views(split, preprocess, render_cfg),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=init_worker,
    )
    task = progress.add_task(split.role, name=f"эмбеддинги {split.role}", total=len(split))
    out: np.ndarray | None = None
    started = time.perf_counter()
    for images, rows in loader:
        features = F.normalize(visual(images.to(device=device, dtype=dtype, non_blocking=True)).float(), dim=-1)
        if out is None:
            out = np.empty((len(split), features.shape[1]), np.float32)
        out[rows.numpy()] = features.cpu().numpy()
        progress.update(task, advance=len(rows))
    if out is None:
        raise click.ClickException(f"в сплите {split.role} не осталось картинок")
    return out, time.perf_counter() - started
