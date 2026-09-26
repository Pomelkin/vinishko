"""Устройство и бэкенд инференса энкодера.

Переменная окружения VIS_SEARCHER_DEV задаёт устройство как в torch: cpu либо cuda:<индекс>. Без неё берётся cuda:0, если CUDA доступна,
иначе cpu. На CUDA граф исполняет TensorRT, в bfloat16 на картах с его аппаратной поддержкой (Ampere и новее), иначе во float32;
на CPU — OpenVINO во float32.
"""

import os
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Literal

import torch

ENV_DEVICE = "VIS_SEARCHER_DEV"
Backend = Literal["tensorrt", "openvino"]
Precision = Literal["bf16", "fp32"]


@dataclass(frozen=True)
class Device:
    """Куда и чем считать энкодер."""

    torch_device: torch.device
    backend: Backend
    precision: Precision

    def __str__(self) -> str:
        return f"{self.torch_device}, {self.backend}, {self.precision}"


def resolve_device(spec: str | None = None) -> Device:
    """Устройство по строке spec, иначе по VIS_SEARCHER_DEV, иначе автоматически; заданное проверяется на доступность."""
    spec = spec or os.environ.get(ENV_DEVICE)
    if spec is None:
        spec = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(spec)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"{ENV_DEVICE}={spec}: CUDA недоступна")
        index = device.index or 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"{ENV_DEVICE}={spec}: видеокарт всего {torch.cuda.device_count()}"
            )
        if find_spec("tensorrt") is None:
            raise RuntimeError(
                "на CUDA граф исполняет TensorRT: нужен пакет tensorrt, группа flash-inference"
            )
        precision: Precision = (
            "bf16" if torch.cuda.get_device_capability(index) >= (8, 0) else "fp32"
        )
        return Device(torch.device("cuda", index), "tensorrt", precision)
    if device.type == "cpu":
        if find_spec("openvino") is None:
            raise RuntimeError(
                "на CPU граф исполняет OpenVINO: нужен пакет openvino, группа cpu-inference"
            )
        return Device(device, "openvino", "fp32")
    raise ValueError(f"{ENV_DEVICE}={spec}: ожидается cpu либо cuda:<индекс>")
