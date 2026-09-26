import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from scripts.bench_common.data import Split
from scripts.bench_common.data import Views
from scripts.bench_common.data import init_worker
from scripts.bench_common.parallel import Advance
from scripts.bench_common.parallel import EmbedTask
from vis_seacher_training.data.augmentations import resize_to_fit

if TYPE_CHECKING:
    from onnxruntime import InferenceSession

MODEL = "DinoV3ForWine"
PREPROCESS_NAME = "preprocess.json"
"""Лежит рядом с model.onnx после экспорта: размер входа и цвет полей, с которыми модель училась."""
INPUT_NAME, OUTPUT_NAME = "images", "embeddings"


def read_contract(
    onnx: Path, input_size: tuple[int, int] | None, fill: tuple[int, int, int]
) -> tuple[tuple[int, int], tuple[int, int, int]]:
    """Размер входа и цвет полей: из preprocess.json рядом с весами, если он есть; --input-size имеет приоритет, цвет полей — из конфига нормализации."""
    contract = onnx.with_name(PREPROCESS_NAME)
    if contract.is_file():
        saved = json.loads(contract.read_text(encoding="utf-8"))
        input_size = input_size or (
            int(saved["input_size"][0]),
            int(saved["input_size"][1]),
        )
        red, green, blue = (int(v) for v in saved.get("pad_color", fill))
        fill = (red, green, blue)
    return input_size or (512, 512), fill


class Preprocess:
    """Вход модели, как его готовит пайплайн: кроп вписывается в input_size с сохранением пропорций, поля цвета заливки фона по центру, float32 в 0…255, RGB, CHW.

    Нормировка ImageNet вшита в граф ONNX, здесь её нет. Тот же путь, что dataset.to_tensor обучения без аугментаций.
    Класс, а не замыкание: воркеры даталоадера стартуют через spawn и получают предобработку пиклом.
    """

    def __init__(self, input_size: tuple[int, int], fill: tuple[int, int, int]) -> None:
        self.input_size, self.fill = input_size, fill

    def __call__(self, img: Image.Image) -> torch.Tensor:
        image = resize_to_fit(np.asarray(img.convert("RGB")), self.input_size)
        canvas = np.empty((*self.input_size, 3), np.uint8)
        canvas[:] = self.fill
        top, left = (
            round((self.input_size[0] - image.shape[0]) * 0.5),
            round((self.input_size[1] - image.shape[1]) * 0.5),
        )
        canvas[top : top + image.shape[0], left : left + image.shape[1]] = image
        return torch.from_numpy(canvas).permute(2, 0, 1).float()


def make_session(onnx: Path, device: torch.device) -> "InferenceSession":
    """Сессия onnxruntime на устройстве воркера. На CUDA нужна сборка onnxruntime-gpu; torch импортирован раньше и уже поднял библиотеки CUDA и cuDNN из своих колёс."""
    import onnxruntime as ort

    if device.type == "cuda":
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                f"в onnxruntime {ort.__version__} нет CUDAExecutionProvider: нужна сборка onnxruntime-gpu"
            )
        providers: list = [
            ("CUDAExecutionProvider", {"device_id": device.index or 0}),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx), sess_options=options, providers=providers)
    if device.type == "cuda" and session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            f"onnxruntime не смог поднять CUDA на {device}: провайдеры сессии {session.get_providers()}"
        )
    return session


BACKENDS = ["onnxruntime", "tensorrt"]


class OnnxRuntimeRunner:
    """Батч float32 CHW → эмбеддинги через сессию onnxruntime."""

    def __init__(self, onnx: Path, device: torch.device) -> None:
        self.session = make_session(onnx, device)
        self.description = f"onnxruntime, {self.session.get_providers()[0]}"

    def __call__(self, images: torch.Tensor) -> np.ndarray:
        (vectors,) = self.session.run([OUTPUT_NAME], {INPUT_NAME: images.numpy()})
        return vectors


Runner = Callable[[torch.Tensor], np.ndarray]


def make_runner(
    backend: str,
    onnx: Path,
    device: torch.device,
    batch_size: int,
    input_size: tuple[int, int],
) -> Runner:
    """Исполнитель графа: onnxruntime с провайдером CUDA либо CPU, или TensorRT — engine собирается один раз на карту и кэшируется рядом с ONNX."""
    if backend == "tensorrt":
        if device.type != "cuda":
            raise RuntimeError(
                "TensorRT работает только на CUDA: укажите --device cuda:<индекс>"
            )
        from vinishko.pipeline.steps.vis_searcher.backends import TensorRTRunner

        return TensorRTRunner(onnx, device, max_batch=batch_size, input_size=input_size)
    return OnnxRuntimeRunner(onnx, device)


def embed(
    runner: Runner,
    preprocess: Preprocess,
    split: Split,
    render_cfg: dict,
    batch_size: int,
    workers: int,
    advance: Advance,
) -> np.ndarray:
    """L2-нормированные эмбеддинги картинок сплита в float32. Граф уже нормирует выход; повторная нормировка — страховка от численного дрейфа."""
    loader = DataLoader(
        Views(split, preprocess, render_cfg),
        batch_size=batch_size,
        num_workers=workers,
        worker_init_fn=init_worker,
    )
    batches = []
    for images, rows in loader:  # без перемешивания батчи идут в порядке сплита
        vectors = runner(images)
        batches.append(
            vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        )
        advance(len(rows))
    return np.concatenate(batches).astype(np.float32)


class DinoWorker:
    """Воркер одного устройства: ONNX-граф DinoV3ForWine с вшитой нормировкой через onnxruntime либо TensorRT, по задаче пишет эмбеддинги своей части сплита в npy."""

    def __init__(
        self,
        device: torch.device,
        onnx: Path,
        backend: str,
        input_size: tuple[int, int],
        fill: tuple[int, int, int],
        render_cfg: dict,
        batch_size: int,
        workers: int,
    ) -> None:
        self.render_cfg, self.batch_size, self.workers = render_cfg, batch_size, workers
        self.runner = make_runner(backend, onnx, device, batch_size, input_size)
        self.preprocess = Preprocess(input_size, fill)
        # dtype нужен пулу устройств, чтобы не смешивать разные; точность графа задана самим ONNX и одна на всех
        self.info = {
            "dtype": "torch.float32",
            "вход": f"{input_size[0]}×{input_size[1]}, поля {fill}",
            "provider": getattr(self.runner, "description", backend),
        }

    def __call__(self, task: EmbedTask, advance: Advance) -> None:
        vectors = embed(
            self.runner,
            self.preprocess,
            task.split,
            self.render_cfg,
            self.batch_size,
            self.workers,
            advance,
        )
        with task.path.open("wb") as f:
            np.save(f, vectors)
