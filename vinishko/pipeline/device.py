"""Устройство torch для шага пайплайна: явно, из переменной окружения шага либо автоматически."""

import os
import re

import torch


def resolve_torch_device(env: str, spec: str | None = None) -> torch.device:
    """cpu либо cuda:<индекс>: из spec, иначе из переменной окружения env, иначе cuda:0 при доступной CUDA, иначе cpu.

    Заданное проверяется: нет CUDA или такой карты — ошибка, а не тихий откат на другое устройство.
    """
    spec = spec or os.environ.get(env)
    if spec is None:
        return (
            torch.device("cuda", 0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    match = re.fullmatch(r"cpu|cuda(?::(\d+))?", spec.strip())
    if match is None:
        raise ValueError(f"{env}={spec}: ожидается cpu либо cuda:<индекс>")
    if match.group(0) == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{env}={spec}: CUDA недоступна")
    index = int(match.group(1) or 0)
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"{env}={spec}: видеокарт всего {torch.cuda.device_count()}")
    return torch.device("cuda", index)
