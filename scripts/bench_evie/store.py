from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np


@dataclass
class TokenStore:
    """Мультивекторные эмбеддинги сплита на диске: токены всех картинок подряд в одном файле float16.

    У картинок разное число токенов, поэтому хранится плоская матрица (всего токенов, dim) и длины;
    в память она не читается, а отображается: галерея val на 128 измерениях — около 6 ГБ, на 2048 — под 90 ГБ.
    """

    path: Path
    dim: int
    lengths: np.ndarray
    """Число токенов каждой картинки, в порядке сплита."""

    def __len__(self) -> int:
        return len(self.lengths)

    @cached_property
    def offsets(self) -> np.ndarray:
        """Номер первого токена каждой картинки; последний элемент — общее число токенов."""
        return np.r_[0, np.cumsum(self.lengths)]

    @cached_property
    def tokens(self) -> np.memmap:
        """Все токены сплита, только на чтение."""
        return np.memmap(self.path, np.float16, "r", shape=(int(self.offsets[-1]), self.dim))

    @property
    def nbytes(self) -> int:
        """Размер файла."""
        return int(self.offsets[-1]) * self.dim * 2

    def padded(self, rows: np.ndarray) -> np.ndarray:
        """Батч (картинки, максимум токенов, dim), добитый нулями: MaxSim пайплайна рассчитан на нулевые векторы вместо маски."""
        out = np.zeros((len(rows), int(self.lengths[rows].max()), self.dim), np.float16)
        for n, row in enumerate(rows):
            out[n, : self.lengths[row]] = self.tokens[self.offsets[row] : self.offsets[row + 1]]
        return out

    def blocks(self, max_tokens: int) -> list[np.ndarray]:
        """Картинки подряд, разбитые на блоки так, чтобы добитый нулями блок не превышал max_tokens токенов."""
        blocks: list[np.ndarray] = []
        start, longest = 0, 0
        for row, length in enumerate(self.lengths):
            longest = max(longest, int(length))
            if row > start and (row - start + 1) * longest > max_tokens:
                blocks.append(np.arange(start, row))
                start, longest = row, int(length)
        blocks.append(np.arange(start, len(self)))
        return blocks
