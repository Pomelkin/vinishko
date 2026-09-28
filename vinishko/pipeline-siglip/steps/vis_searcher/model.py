"""ONNX SigLIP2: существующие TensorRT/OpenVINO и bilinear resize экспорта."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from vinishko.pipeline.steps.vis_searcher.model import Encoder as BaseEncoder
from vinishko.pipeline.steps.vis_searcher.model import ModelFiles, Preprocess as BasePreprocess
from vinishko.pipeline.steps.vis_searcher.device import Device


class Preprocess(BasePreprocess):
    """RGB uint8 → bilinear squash → float32 NCHW 0…255; нормировка уже в ONNX."""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
            raise ValueError("ожидается непустой uint8 RGB HWC")
        resized = Image.fromarray(image).resize(self.input_size[::-1], Image.Resampling.BILINEAR)
        return np.asarray(resized).transpose(2, 0, 1).astype(np.float32)


class Encoder(BaseEncoder):
    """Сохраняет батчи, L2-нормировку, проверку выхода и кэш engine оригинала."""

    def __init__(self, files: ModelFiles, device: Device, max_batch: int,
                 cache_dir: Path, cpu_max_batch: int = 1) -> None:
        contract = json.loads((files.root / "preprocess.json").read_text(encoding="utf-8"))
        if files.resize != "squash" or contract.get("interpolation") != "bilinear":
            raise ValueError("SigLIP ожидает preprocess.json с resize=squash и interpolation=bilinear")
        super().__init__(files, device, max_batch, cache_dir, cpu_max_batch)
        self.preprocess = Preprocess(files.input_size, files.pad_color, files.resize)
