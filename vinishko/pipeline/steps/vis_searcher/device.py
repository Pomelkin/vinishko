"""Устройство и бэкенд инференса энкодера.

Переменная окружения VIS_SEARCHER_DEV задаёт устройство как в torch: cpu либо cuda:<индекс>. Без неё берётся cuda:0, если CUDA доступна,
иначе cpu. На CUDA граф исполняет TensorRT, в bfloat16 на картах с его аппаратной поддержкой (Ampere и новее), иначе во float32;
на CPU — OpenVINO во float32.
"""

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Literal

import torch

from vinishko.pipeline.device import resolve_torch_device

ENV_DEVICE = "VIS_SEARCHER_DEV"
Backend = Literal["tensorrt", "openvino"]
Precision = Literal["bf16", "fp32"]


@dataclass(frozen=True, slots=True)
class Device:
    """Куда и чем считать энкодер."""

    torch_device: torch.device
    backend: Backend
    precision: Precision

    def __str__(self) -> str:
        return f"{self.torch_device}, {self.backend}, {self.precision}"


def resolve_device(spec: str | None = None) -> Device:
    """Устройство по строке spec, иначе по VIS_SEARCHER_DEV, иначе автоматически; заданное проверяется на доступность."""
    device = resolve_torch_device(ENV_DEVICE, spec)
    if device.type == "cuda":
        if find_spec("tensorrt") is None:
            raise RuntimeError(
                "на CUDA граф исполняет TensorRT: нужен пакет tensorrt, группа flash-inference"
            )
        precision: Precision = (
            "bf16"
            if torch.cuda.get_device_capability(device.index) >= (8, 0)
            else "fp32"
        )
        return Device(device, "tensorrt", precision)
    if find_spec("openvino") is None:
        raise RuntimeError(
            "на CPU граф исполняет OpenVINO: нужен пакет openvino, группа cpu-inference"
        )
    return Device(device, "openvino", "fp32")
