"""Модель визуального поиска: файлы экспорта с Hugging Face либо из локальной директории, предобработка по preprocess.json и энкодер."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from huggingface_hub import snapshot_download

from vinishko.pipeline.steps.vis_searcher.backends import OpenVinoRunner, Runner, TensorRTRunner
from vinishko.pipeline.steps.vis_searcher.device import Device, Precision

ONNX_FP32, PREPROCESS_NAME, CONFIG_NAME = "model.onnx", "preprocess.json", "config.json"
Resize = Literal["pad", "squash"]
"""pad — вписать с сохранением пропорций и дополнить полями (DINOv3 для вина); squash — растянуть до размера входа без полей (TULIP, SigLIP)."""
LOCAL_PREFIX = "local-"


@dataclass(frozen=True, slots=True)
class ModelFiles:
    """Контракт модели: откуда взята, ревизия и как готовить вход."""

    repo: str
    """Репозиторий Hugging Face либо локальная директория экспорта."""
    revision: str
    """Хэш коммита репозитория, у директории — хэш контракта и графа: коллекция помнит его и с другой ревизией не работает."""
    root: Path
    """Снимок репозитория в кэше huggingface_hub либо сама директория."""
    onnx_name: str
    """Имя графа нужной точности; сам файл докачивает download_onnx, чтобы проверки перед стартом не ждали сотен мегабайт."""
    precision: Precision
    input_size: tuple[int, int]
    pad_color: tuple[int, int, int]
    resize: Resize
    embed_dim: int
    patch_size: int


def fetch_model(repo: str, precision: Precision, revision: str | None = None) -> ModelFiles:
    """Контракт модели: config.json и preprocess.json из репозитория Hugging Face через кэш huggingface_hub либо из локальной директории.

    Граф здесь не качается. Имя bf16-графа берётся из preprocess.json, который пишет экспорт. Поля resize в экспортах до 2026-09-26 нет:
    они все вписывали с полями, поэтому без поля — pad.
    """
    local = Path(repo).expanduser()
    root = local if local.is_dir() else Path(snapshot_download(repo, revision=revision, allow_patterns=[PREPROCESS_NAME, CONFIG_NAME]))
    contract = json.loads((root / PREPROCESS_NAME).read_text(encoding="utf-8"))
    if precision == "bf16":
        if not contract.get("onnx_bf16"):
            raise RuntimeError(f"в {repo} нет bf16-графа: preprocess.json без поля onnx_bf16")
        onnx_name = contract["onnx_bf16"]
    else:
        onnx_name = ONNX_FP32
    config = json.loads((root / CONFIG_NAME).read_text(encoding="utf-8"))
    red, green, blue = (int(v) for v in contract["pad_color"])
    resize = contract.get("resize", "pad")
    if resize not in ("pad", "squash"):
        raise RuntimeError(f"в {repo} preprocess.json: resize={resize!r}, ожидается pad либо squash")
    return ModelFiles(
        repo=repo,
        revision=local_revision(root, onnx_name) if local.is_dir() else root.name,
        root=root,
        onnx_name=onnx_name,
        precision=precision,
        input_size=(int(contract["input_size"][0]), int(contract["input_size"][1])),
        pad_color=(red, green, blue),
        resize=resize,
        embed_dim=int(config["embed_dim"]),
        patch_size=int(contract["patch_size"]),
    )


def local_revision(root: Path, onnx_name: str) -> str:
    """Ревизия локальной директории: sha256 от config.json, preprocess.json и содержимого графа.

    Граф читается целиком, несколько секунд на гигабайт: у переэкспорта той же архитектуры в ту же директорию размеры файлов совпадают,
    и ревизия по размеру подсунула бы коллекции и кэшу engine старую модель.
    """
    digest = hashlib.sha256()
    for name in (CONFIG_NAME, PREPROCESS_NAME):
        digest.update((root / name).read_bytes())
    onnx = root / onnx_name
    if not onnx.is_file():
        raise RuntimeError(f"в {root} нет {onnx_name}")
    with onnx.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return LOCAL_PREFIX + digest.hexdigest()[:16]


def download_onnx(files: ModelFiles) -> Path:
    """Граф нужной точности: у локальной директории он должен лежать в ней, у репозитория докачивается в тот же снимок кэша."""
    if files.revision.startswith(LOCAL_PREFIX):
        if not (files.root / files.onnx_name).is_file():
            raise RuntimeError(f"в {files.root} нет {files.onnx_name}")
        return files.root / files.onnx_name
    root = Path(snapshot_download(files.repo, revision=files.revision, allow_patterns=[files.onnx_name, files.onnx_name + ".data"]))
    if root != files.root or not (root / files.onnx_name).is_file():
        raise RuntimeError(f"в {files.repo}@{files.revision[:8]} нет {files.onnx_name}")
    return root / files.onnx_name


def fit_size(height: int, width: int, target: tuple[int, int]) -> tuple[int, int]:
    """Размер, с которым картинка вписывается в target с сохранением пропорций."""
    scale = min(target[0] / height, target[1] / width)
    return min(target[0], max(1, round(height * scale))), min(target[1], max(1, round(width * scale)))


def resize_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Ресайз к размеру: area при уменьшении, cubic при увеличении — как готовился вход на обучении."""
    interpolation = cv2.INTER_AREA if height < image.shape[0] else cv2.INTER_CUBIC
    return cv2.resize(image, (width, height), interpolation=interpolation)


def resize_to_fit(image: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Вписывает картинку в target с сохранением пропорций."""
    return resize_image(image, *fit_size(image.shape[0], image.shape[1], target))


class Preprocess:
    """Вход модели по preprocess.json: float32 в 0…255, RGB, CHW размера input_size.

    resize=pad — кроп вписывается с сохранением пропорций, поля цвета pad_color по центру; squash — растягивается до input_size без полей.
    """

    def __init__(self, input_size: tuple[int, int], pad_color: tuple[int, int, int], resize: Resize = "pad") -> None:
        self.input_size, self.pad_color, self.resize = input_size, pad_color, resize

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """uint8 RGB HWC любого размера → float32 CHW размера входа."""
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"ожидается uint8 RGB HWC, пришло {image.dtype} {image.shape}")
        if self.resize == "squash":
            return resize_image(image, *self.input_size).transpose(2, 0, 1).astype(np.float32)
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

    def __init__(self, files: ModelFiles, device: Device, max_batch: int, cache_dir: Path, cpu_max_batch: int = 1) -> None:
        self.files, self.device = files, device
        self.max_batch = min(max_batch, cpu_max_batch) if device.backend == "openvino" else max_batch
        self.preprocess = Preprocess(files.input_size, files.pad_color, files.resize)
        onnx = download_onnx(files)
        self.runner: Runner
        if device.backend == "tensorrt":
            engines = cache_dir / "engines" / files.repo.replace("/", "__").strip("_") / files.revision
            self.runner = TensorRTRunner(onnx, device.torch_device, max_batch, files.input_size, engines)
        else:
            self.runner = OpenVinoRunner(onnx)
        probe = self.runner(np.zeros((1, 3, *files.input_size), np.float32))
        if probe.shape != (1, files.embed_dim):
            raise RuntimeError(f"граф отдаёт {probe.shape}, а config.json обещает embed_dim={files.embed_dim}")

    @property
    def embed_dim(self) -> int:
        """Размерность эмбеддинга."""
        return self.files.embed_dim

    @property
    def description(self) -> str:
        """Модель, вход, устройство и бэкенд одной строкой для логов."""
        files = self.files
        return f"{files.repo}@{files.revision[:14]} {files.onnx_name}, вход {files.input_size[0]}×{files.input_size[1]} {files.resize}, {self.device}, {self.runner.description}"

    def __call__(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Эмбеддинги float32 (n, embed_dim), L2-нормированные; граф уже нормирует выход, повторная нормировка страхует от численного дрейфа."""
        if not images:
            return np.empty((0, self.embed_dim), np.float32)
        chunks = []
        for start in range(0, len(images), self.max_batch):
            vectors = self.runner(self.preprocess.batch(images[start : start + self.max_batch]))
            chunks.append(vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12))
        return np.concatenate(chunks).astype(np.float32)
