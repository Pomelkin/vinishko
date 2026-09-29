"""Второй уровень: выбор одной позиции каталога среди кандидатов поиска — внутри каждой группы near-duplicates, затем среди лучших позиций групп."""

from .resolve import NearDuplicateError
from .resolve import NearDuplicateReason
from .resolve import NearDuplicateResolver


__all__ = ["NearDuplicateError", "NearDuplicateReason", "NearDuplicateResolver"]
