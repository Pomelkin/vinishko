"""Общий экспорт визуальных энкодеров в формат моделей визуального поиска: model.onnx, model.bf16.onnx, config.json, preprocess.json.

Контракт тот же, что у экспорта DinoV3ForWine: вход float32 RGB NCHW 0…255, нормировка внутри графа, выход L2-нормированные эмбеддинги,
динамический батч. Конкретные модели, TULIP и SigLIP2, лишь подставляют свою башню и препроцессинг.
"""

import copy
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from kostyl.utils import setup_logger
from torch import nn
from torch.export import Dim

from vis_seacher_training.export import BF16_MIN_COSINE, DUMMY_BATCH, ONNX_BF16_NAME, ONNX_NAME, OPSET

logger = setup_logger(fmt="detailed")

RESIZES = ("squash", "pad")


class EncoderForExport(nn.Module):
    """Башня под контракт поиска: вход 0…255, нормировка в графе, на выходе L2-нормированные эмбеддинги float32.

    embedder — модуль, отдающий эмбеддинги по нормированному NCHW-входу; compute_dtype — тип весов и вычислений: float32 для onnxruntime,
    bfloat16 для TensorRT.
    """

    def __init__(self, embedder: nn.Module, mean: tuple[float, ...], std: tuple[float, ...], compute_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.embedder = embedder.to(compute_dtype).eval()
        self.compute_dtype = compute_dtype
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixel_values = (images / 255.0 - self.mean) / self.std  # ty: ignore[unsupported-operator]
        return F.normalize(self.embedder(pixel_values.to(self.compute_dtype)).float(), dim=-1)


@dataclass(frozen=True, slots=True)
class Contract:
    """Как поиск готовит вход модели."""

    input_size: tuple[int, int]
    patch_size: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    resize: str
    """squash — растянуть до входа без полей; pad — вписать с полями цвета pad_color."""
    pad_color: tuple[int, int, int]
    interpolation: str = "area_cubic"
    """area_cubic — cv2 как на обучении DINO; bilinear и bicubic — PIL с антиалиасингом, как препроцессоры transformers и open_clip."""


def export_onnx(encoder: EncoderForExport, path: Path, input_size: tuple[int, int]) -> torch.Tensor:
    """ONNX одним файлом; динамический только батч: у башен с фиксированным входом позиционные эмбеддинги под свой размер."""
    images = torch.rand(DUMMY_BATCH, 3, *input_size) * 255
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            (images,),
            path,
            dynamo=True,
            external_data=False,
            opset_version=OPSET,
            input_names=["images"],
            output_names=["embeddings"],
            dynamic_shapes={"images": {0: Dim("batch")}},
        )
    inline_external_data(path)
    logger.info(f"ONNX записан в {path}: {path.stat().st_size / 2**20:.0f} МБ, вход {tuple(images.shape)}, ось batch динамическая")
    return images


def inline_external_data(path: Path) -> None:
    """Граф одним файлом: экспортёр выносит веса в <имя>.data, когда считает граф большим, а контракт поиска ждёт один файл до 2 ГБ."""
    data = path.with_name(path.name + ".data")
    if not data.is_file():
        return
    model = onnx.load(str(path), load_external_data=True)
    onnx.save_model(model, str(path))
    data.unlink()
    logger.info(f"веса из {data.name} вложены в {path.name}")


def verify_onnx(encoder: EncoderForExport, path: Path, images: torch.Tensor) -> float:
    """Сверка с PyTorch на трассировочном входе и на батче другого размера: наибольшее расхождение по модулю."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    other = torch.rand(DUMMY_BATCH + 1, 3, *images.shape[2:]) * 255
    worst = 0.0
    with torch.no_grad():
        for batch in (images, other):
            want = encoder(batch).numpy()
            (got,) = session.run(["embeddings"], {"images": batch.numpy()})
            diff = float(np.abs(want - got).max())
            worst = max(worst, diff)
            logger.info(f"вход {tuple(batch.shape)}: расхождение с PyTorch {diff:.2e}")
    return worst


def verify_tensorrt(reference: EncoderForExport, path: Path, images: torch.Tensor, input_size: tuple[int, int]) -> float | None:
    """Сверка bf16-графа через TensorRT с PyTorch float32: наименьший косинус по батчу; None — TensorRT не установлен."""
    from vinishko.pipeline.steps.vis_searcher.backends import TensorRTRunner, tensorrt_available

    if not tensorrt_available():
        logger.warning(f"{path.name} не сверен: нет tensorrt либо CUDA")
        return None
    with tempfile.TemporaryDirectory(prefix="trt-verify-") as tmp:  # engine сверки одноразовый: не в директорию экспорта, которая уезжает на HF
        runner = TensorRTRunner(path, torch.device("cuda", 0), max_batch=images.shape[0], input_size=input_size, cache_dir=Path(tmp))
        with torch.no_grad():
            want = reference(images).numpy()
        got = runner(images)
    cosine = float((want * got).sum(1).min())
    logger.info(f"{path.name} через TensorRT: наименьший косинус с PyTorch float32 {cosine:.5f}")
    return cosine


def export_bundle(embedder: nn.Module, output: Path, contract: Contract, config: dict, bf16: bool) -> None:
    """Всё, что нужно поиску, в одну директорию: графы float32 и bf16 со сверками, config.json с embed_dim и preprocess.json."""
    output.mkdir(parents=True, exist_ok=True)
    encoder = EncoderForExport(copy.deepcopy(embedder), contract.mean, contract.std)
    with torch.no_grad():
        embed_dim = int(encoder(torch.zeros(1, 3, *contract.input_size)).shape[1])
    logger.info(f"{config.get('architecture')}: вход {contract.input_size}, патч {contract.patch_size}, эмбеддинг {embed_dim}, mean {contract.mean}, std {contract.std}")
    images = export_onnx(encoder, output / ONNX_NAME, contract.input_size)
    worst = verify_onnx(encoder, output / ONNX_NAME, images)
    if worst > 1e-3:
        raise RuntimeError(f"ONNX расходится с PyTorch на {worst:.2e}: экспорт не годится")
    trt_cosine = None
    if bf16:
        half = EncoderForExport(copy.deepcopy(embedder), contract.mean, contract.std, torch.bfloat16)
        export_onnx(half, output / ONNX_BF16_NAME, contract.input_size)
        trt_cosine = verify_tensorrt(encoder, output / ONNX_BF16_NAME, images, contract.input_size)
        if trt_cosine is not None and trt_cosine < BF16_MIN_COSINE:
            raise RuntimeError(f"bf16-граф через TensorRT расходится с PyTorch: косинус {trt_cosine:.4f} < {BF16_MIN_COSINE}")
    (output / "config.json").write_text(
        json.dumps({**config, "embed_dim": embed_dim, "image_size": list(contract.input_size), "patch_size": contract.patch_size}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    preprocess = {
        "input": "images: float32, RGB, NCHW, значения 0…255; нормировка внутри графа",
        "output": "embeddings: float32 (batch, embed_dim), L2-нормированные",
        "input_size": list(contract.input_size),
        "patch_size": contract.patch_size,
        "pad_color": list(contract.pad_color),
        "resize": contract.resize,
        "interpolation": contract.interpolation,
        "preprocess": "растянуть кроп до input_size без полей" if contract.resize == "squash" else "вписать кроп в input_size с сохранением пропорций, дополнить pad_color по центру",
        "mean": list(contract.mean),
        "std": list(contract.std),
        "onnx_max_abs_diff_vs_torch": worst,
        "onnx_bf16": ONNX_BF16_NAME if bf16 else None,
        "onnx_bf16_min_cosine_vs_torch_tensorrt": trt_cosine,
    }
    (output / "preprocess.json").write_text(json.dumps(preprocess, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info(f"готово: {output}")
