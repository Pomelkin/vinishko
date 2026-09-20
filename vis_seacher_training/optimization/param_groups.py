from typing import Any

from kostyl.utils import setup_logger
from torch import nn

logger = setup_logger(fmt="detailed")

NO_DECAY_KEYWORDS = ("cls_token", "register_tokens", "mask_token", "loss_fn")
"""Без weight decay, помимо всех одномерных параметров: токены бэкбона и центры ArcFace — у центров важно только направление."""


def create_param_groups(module: nn.Module, lr: float, weight_decay: float, backbone_prefix: str, backbone_lr_multiplier: float | None) -> list[dict[str, Any]]:
    """Группы параметров оптимизатора.

    Weight decay не получают смещения, нормировки, layer scale и степень GeM, то есть все одномерные параметры, и то, что перечислено
    в NO_DECAY_KEYWORDS. Параметрам бэкбона проставляется lr_multiplier: планировщик kostyl умножает на него скорость обучения группы.
    """
    groups: dict[tuple[float, float], dict[str, Any]] = {}
    slowed = 0
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        decay = 0.0 if param.ndim <= 1 or any(keyword in name for keyword in NO_DECAY_KEYWORDS) else weight_decay
        multiplier = backbone_lr_multiplier if backbone_lr_multiplier is not None and name.startswith(backbone_prefix) else 1.0
        slowed += multiplier != 1.0
        group = groups.setdefault((decay, multiplier), {"params": [], "lr": lr * multiplier, "weight_decay": decay, "lr_multiplier": multiplier})
        group["params"].append(param)
    logger.log_rank_zero("INFO", f"Групп параметров: {len(groups)}; с множителем lr {backbone_lr_multiplier}: {slowed} тензоров бэкбона")
    return list(groups.values())
