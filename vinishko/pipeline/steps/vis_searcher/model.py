"""Модель визуального поиска: файлы экспорта DinoV3ForWine с Hugging Face, предобработка по preprocess.json и энкодер."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import snapshot_download

from vinishko.pipeline.steps.vis_searcher.backends import (
    OpenVinoRunner,
    Runner,
    TensorRTRunner,
)
from vinishko.pipeline.steps.vis_searcher.device import Device, Precision

ONNX_FP32, PREPROCESS_NAME, CONFIG_NAME = "model.onnx", "preprocess.json", "config.json"


@dataclass(frozen=True, slots=True)
class ModelFiles:
    """Что скачано с Hugging Face и контракт входа из preprocess.json."""

    repo: str
    revision: str
    """Хэш коммита репозитория: коллекция помнит его и с другой ревизией не работает."""
    root: Path
    """Снимок репозитория в кэше huggingface_hub."""
    onnx_name: str
    """Имя графа нужной точности; сам файл докачивает download_onnx, чтобы проверки перед стартом не ждали сотен мегабайт."""
    precision: Precision
    input_size: tuple[int, int]
    pad_color: tuple[int, int, int]
    embed_dim: int
    patch_size: int


def fetch_model(
    repo: str, precision: Precision, revision: str | None = None
) -> ModelFiles:
    """Контракт модели из репозитория Hugging Face через кэш huggingface_hub: config.json и preprocess.json, без графа.

    Имя bf16-графа берётся из preprocess.json, который пишет экспорт обучения; ревизия — хэш коммита снимка.
    """
    root = Path(
        snapshot_download(
            repo, revision=revision, allow_patterns=[PREPROCESS_NAME, CONFIG_NAME]
        )
    )
    contract = json.loads((root / PREPROCESS_NAME).read_text(encoding="utf-8"))
    if precision == "bf16":
        if "onnx_bf16" not in contract:
            raise RuntimeError(
                f"в {repo} нет bf16-графа: preprocess.json без поля onnx_bf16"
            )
        onnx_name = contract["onnx_bf16"]
    else:
        onnx_name = ONNX_FP32
    config = json.loads((root / CONFIG_NAME).read_text(encoding="utf-8"))
    red, green, blue = (int(v) for v in contract["pad_color"])
    return ModelFiles(
        repo=repo,
        revision=root.name,
        root=root,
        onnx_name=onnx_name,
        precision=precision,
        input_size=(int(contract["input_size"][0]), int(contract["input_size"][1])),
        pad_color=(red, green, blue),
        embed_dim=int(config["embed_dim"]),
        patch_size=int(contract["patch_size"]),
    )


def download_onnx(files: ModelFiles) -> Path:
    """Граф нужной точности в тот же снимок кэша; повторный вызов ничего не качает."""
    root = Path(
        snapshot_download(
            files.repo, revision=files.revision, allow_patterns=[files.onnx_name]
        )
    )
    if root != files.root or not (root / files.onnx_name).is_file():
        raise RuntimeError(f"в {files.repo}@{files.revision[:8]} нет {files.onnx_name}")
    return root / files.onnx_name


def fit_size(height: int, width: int, target: tuple[int, int]) -> tuple[int, int]:
    """Размер, с которым картинка вписывается в target с сохранением пропорций."""
    scale = min(target[0] / height, target[1] / width)
    return min(target[0], max(1, round(height * scale))), min(
        target[1], max(1, round(width * scale))
    )


def resize_to_fit(image: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Вписывает картинку в target: area при уменьшении, cubic при увеличении — как готовился вход на обучении."""
    height, width = fit_size(image.shape[0], image.shape[1], target)
    interpolation = cv2.INTER_AREA if height < image.shape[0] else cv2.INTER_CUBIC
    return cv2.resize(image, (width, height), interpolation=interpolation)


class Preprocess:
    """Вход модели по preprocess.json: кроп вписывается в input_size, поля цвета pad_color по центру, float32 в 0…255, RGB, CHW."""

    def __init__(
        self, input_size: tuple[int, int], pad_color: tuple[int, int, int]
    ) -> None:
        self.input_size, self.pad_color = input_size, pad_color

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """uint8 RGB HWC любого размера → float32 CHW размера входа."""
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"ожидается uint8 RGB HWC, пришло {image.dtype} {image.shape}"
            )
        fitted = resize_to_fit(image, self.input_size)
        canvas = np.empty((*self.input_size, 3), np.uint8)
        canvas[:] = self.pad_color
        top = round((self.input_size[0] - fitted.shape[0]) * 0.5)
        left = round((self.input_size[1] - fitted.shape[1]) * 0.5)
        canvas[top : top + fitted.shape[0], left : left + fitted.shape[1]] = fitted
        return canvas.transpose(2, 0, 1).astype(np.float32)

    def batch(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Батч NCHW."""
        return np.stack([self(image) for image in images])


class Encoder:
    """Кропы бутылок → L2-нормированные эмбеддинги; бэкенд по устройству, батчи не больше max_batch."""

    def __init__(
        self, files: ModelFiles, device: Device, max_batch: int, cache_dir: Path, cpu_max_batch: int = 1
    ) -> None:
        self.files, self.device = files, device
        self.max_batch = min(max_batch, cpu_max_batch) if device.backend == "openvino" else max_batch
        self.preprocess = Preprocess(files.input_size, files.pad_color)
        onnx = download_onnx(files)
        self.runner: Runner
        if device.backend == "tensorrt":
            engines = (
                cache_dir / "engines" / files.repo.replace("/", "__") / files.revision
            )
            self.runner = TensorRTRunner(
                onnx, device.torch_device, max_batch, files.input_size, engines
            )
        else:
            self.runner = OpenVinoRunner(onnx)
        probe = self.runner(np.zeros((1, 3, *files.input_size), np.float32))
        if probe.shape != (1, files.embed_dim):
            raise RuntimeError(
                f"граф отдаёт {probe.shape}, а config.json обещает embed_dim={files.embed_dim}"
            )

    @property
    def embed_dim(self) -> int:
        """Размерность эмбеддинга."""
        return self.files.embed_dim

    @property
    def description(self) -> str:
        """Модель, устройство и бэкенд одной строкой для логов."""
        return f"{self.files.repo}@{self.files.revision[:8]} {self.files.onnx_name}, {self.device}, {self.runner.description}, batch≤{self.max_batch}"

    def __call__(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Эмбеддинги float32 (n, embed_dim), L2-нормированные; граф уже нормирует выход, повторная нормировка страхует от численного дрейфа."""
        if not images:
            return np.empty((0, self.embed_dim), np.float32)
        chunks = []
        for start in range(0, len(images), self.max_batch):
            vectors = self.runner(
                self.preprocess.batch(images[start : start + self.max_batch])
            )
            chunks.append(
                vectors
                / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            )
        return np.concatenate(chunks).astype(np.float32)
