"""Фильтр кандидатов top_n: один slug либо отказ, без групп."""

from .resolve import FilterError
from .resolve import FilterReason
from .resolve import FilterResolver


__all__ = ["FilterError", "FilterReason", "FilterResolver"]
