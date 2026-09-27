"""Второй уровень: выбор одной позиции каталога внутри группы near-duplicates."""

from .resolve import NearDuplicateError
from .resolve import NearDuplicateReason
from .resolve import NearDuplicateResolver


__all__ = ["NearDuplicateError", "NearDuplicateReason", "NearDuplicateResolver"]
