"""Исполнители ONNX-графа DinoV3ForWine: TensorRT на CUDA и OpenVINO на CPU.

Вход обоих — float32 NCHW со значениями 0…255, нормировка ImageNet вшита в граф; выход — float32 (batch, embed_dim).
В TensorRT 11 флагов точности нет: сеть strongly typed, и вычисления идут в типах графа — bf16-экспорт даёт bf16, fp32-экспорт fp32.
onnxruntime bf16-граф не исполняет, поэтому bf16-ONNX — только для TensorRT.
"""

import time
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from kostyl.utils import setup_logger

logger = setup_logger(fmt="detailed")

INPUT_NAME, OUTPUT_NAME = "images", "embeddings"
WORKSPACE_BYTES = 4 << 30


class Runner(Protocol):
    """Батч float32 NCHW → эмбеддинги float32 на CPU."""

    description: str

    def __call__(self, images: np.ndarray | torch.Tensor) -> np.ndarray: ...


def tensorrt_available() -> bool:
    """Установлен ли tensorrt и есть ли CUDA."""
    return find_spec("tensorrt") is not None and torch.cuda.is_available()


def load_tensorrt() -> Any:
    """Модуль tensorrt как Any: он реэкспортирует всё из скомпилированного tensorrt_bindings звёздочкой, и проверка типов его членов не видит."""
    return import_module("tensorrt")


def engine_path(
    onnx: Path,
    device: torch.device,
    max_batch: int,
    input_size: tuple[int, int],
    cache_dir: Path | None = None,
) -> Path:
    """Файл engine: рядом с ONNX либо в cache_dir; имя привязано к модели, видеокарте, версии TensorRT, размеру входа и потолку батча."""
    trt = load_tensorrt()
    gpu = torch.cuda.get_device_name(device).replace(" ", "_")
    name = f"{onnx.stem}.{gpu}.trt{trt.__version__}.{input_size[0]}x{input_size[1]}.b{max_batch}.engine"
    return (cache_dir or onnx.parent) / name


def build_engine(
    onnx: Path, device: torch.device, max_batch: int, input_size: tuple[int, int]
) -> bytes:
    """Сериализованный engine из ONNX для карты device: сеть strongly typed, точность берётся из типов графа, батч динамический до max_batch.

    TensorRT собирает engine под текущее устройство CUDA и его тактики, поэтому устройство выставляется здесь, а не только в раннере.
    """
    torch.cuda.set_device(device)
    trt = load_tensorrt()
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx)):
        errors = [parser.get_error(i).desc() for i in range(parser.num_errors)]
        raise RuntimeError(f"TensorRT не разобрал {onnx}: {errors[:3]}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    profile = builder.create_optimization_profile()
    shape = (3, *input_size)
    profile.set_shape(INPUT_NAME, (1, *shape), (max_batch, *shape), (max_batch, *shape))
    config.add_optimization_profile(profile)
    started = time.time()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TensorRT не собрал engine из {onnx}")
    logger.info(
        f"engine TensorRT собран за {time.time() - started:.0f} с, {plan.nbytes / 2**20:.0f} МБ"
    )
    return bytes(plan)


class TensorRTRunner:
    """Исполнение ONNX через TensorRT: engine собирается один раз на видеокарту и кэшируется; вход и выход — тензоры torch на устройстве.

    Размер входа фиксирован профилем, батч — от 1 до max_batch. Кэш — рядом с ONNX либо в cache_dir.
    """

    def __init__(
        self,
        onnx: Path,
        device: torch.device,
        max_batch: int,
        input_size: tuple[int, int],
        cache_dir: Path | None = None,
    ) -> None:
        trt = load_tensorrt()
        torch.cuda.set_device(device)
        self.device, self.input_size, self.max_batch = device, input_size, max_batch
        cache = engine_path(onnx, device, max_batch, input_size, cache_dir)
        if cache.is_file():
            plan = cache.read_bytes()
            logger.info(f"engine TensorRT из кэша {cache}")
        else:
            logger.info(
                f"engine TensorRT для {onnx.name} на {torch.cuda.get_device_name(device)} собирается, это несколько минут"
            )
            plan = build_engine(onnx, device, max_batch, input_size)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(plan)
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        self.context = self.engine.create_execution_context()
        self.embed_dim = int(self.engine.get_tensor_shape(OUTPUT_NAME)[1])
        self.stream = torch.cuda.Stream(device)
        self.description = f"TensorRT {trt.__version__}, {cache.name}"

    def __call__(self, images: np.ndarray | torch.Tensor) -> np.ndarray:
        """Эмбеддинги батча float32 на CPU."""
        batch = torch.as_tensor(images).to(self.device, non_blocking=True).contiguous()
        if (
            tuple(batch.shape[1:]) != (3, *self.input_size)
            or not 0 < batch.shape[0] <= self.max_batch
        ):
            raise ValueError(
                f"вход {tuple(batch.shape)} не подходит профилю engine: батч 1…{self.max_batch}, размер {self.input_size}"
            )
        out = torch.empty(
            batch.shape[0], self.embed_dim, device=self.device, dtype=torch.float32
        )
        with torch.cuda.stream(self.stream):
            self.context.set_input_shape(INPUT_NAME, tuple(batch.shape))
            self.context.set_tensor_address(INPUT_NAME, batch.data_ptr())
            self.context.set_tensor_address(OUTPUT_NAME, out.data_ptr())
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT: execute_async_v3 вернул ошибку")
        self.stream.synchronize()
        return out.cpu().numpy()


class OpenVinoRunner:
    """Исполнение float32-ONNX через OpenVINO на CPU; формы динамические, батч любой."""

    def __init__(self, onnx: Path) -> None:
        ov = import_module("openvino")
        core = ov.Core()
        self.compiled = core.compile_model(core.read_model(str(onnx)), "CPU")
        self.description = f"OpenVINO {ov.__version__}, CPU"

    def __call__(self, images: np.ndarray | torch.Tensor) -> np.ndarray:
        """Эмбеддинги батча float32."""
        batch = images.numpy() if isinstance(images, torch.Tensor) else images
        return np.asarray(
            self.compiled(np.ascontiguousarray(batch, dtype=np.float32))[0],
            dtype=np.float32,
        )
