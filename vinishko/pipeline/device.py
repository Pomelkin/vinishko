"""Устройство torch для шага пайплайна: переменная окружения шага перекрывает значение из его конфига, auto — выбор по наличию CUDA."""

import os
import re

import torch
from kostyl.utils import setup_logger

logger = setup_logger(fmt="detailed")

AUTO = "auto"


def resolve_torch_device(env: str, configured: str | None = None) -> torch.device:
    """cpu либо cuda:<индекс>: из переменной окружения env, иначе из configured (значение конфига), иначе auto.

    auto, пустое значение или None — cuda:0 при доступной CUDA, иначе cpu. Заданное проверяется: нет CUDA, такой карты
    или строка не cpu/cuda:<индекс> — ошибка с указанием, откуда взято значение, а не тихий откат на другое устройство.
    Перекрытие конфига переменной и выбор auto пишутся в лог: по логу должно быть видно, откуда взялось устройство.
    """
    from_env = os.environ.get(env)
    if from_env:
        source, spec = env, from_env
        logger.info(
            f"{env}={from_env} перекрывает device в конфиге ({configured or AUTO})"
        )
    else:
        source, spec = "device в конфиге", configured
    if spec is None or spec.strip() in ("", AUTO):
        if torch.cuda.is_available():
            logger.info(f"{env}: auto → cuda:0 ({torch.cuda.get_device_name(0)})")
            return torch.device("cuda", 0)
        logger.info(f"{env}: auto → cpu, CUDA недоступна")
        return torch.device("cpu")
    match = re.fullmatch(r"cpu|cuda(?::(\d+))?", spec.strip())
    if match is None:
        raise ValueError(f"{source}={spec}: ожидается auto, cpu либо cuda:<индекс>")
    if match.group(0) == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{source}={spec}: CUDA недоступна")
    index = int(match.group(1) or 0)
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"{source}={spec}: видеокарт всего {torch.cuda.device_count()}"
        )
    return torch.device("cuda", index)
