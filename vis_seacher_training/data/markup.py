import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from kostyl.utils import setup_logger

logger = setup_logger(fmt="detailed")

MARKUP_NAME = "normalization.jsonl"

Span = tuple[int, int]
"""Где в normalization.jsonl лежит строка картинки: смещение и длина в байтах."""


@dataclass(frozen=True)
class Items:
    """Картинки одной роли: имена, строковые метки классов и место разметки нормализации."""

    root: Path
    """Директория датасета: картинки в images, разметка в normalization.jsonl."""
    names: list[str]
    labels: list[str]
    spans: list[Span | None]
    """None — годной бутылки на картинке нет: такая идёт в модель целиком, если её вообще берут."""

    def __len__(self) -> int:
        return len(self.names)

    def path(self, index: int) -> Path:
        """Путь до картинки."""
        return self.root / "images" / self.names[index]

    def take(self, rows: list[int]) -> "Items":
        """Подмножество картинок."""
        return Items(self.root, [self.names[i] for i in rows], [self.labels[i] for i in rows], [self.spans[i] for i in rows])

    def with_bottle(self) -> "Items":
        """Только картинки, где нормализация нашла годную бутылку: остальные пайплайн до энкодера не допускает."""
        return self.take([i for i, span in enumerate(self.spans) if span is not None])

    def sample_classes(self, max_images: int | None, max_per_class: int | None = None) -> "Items":
        """Случайная, но воспроизводимая выборка классов: не больше max_images картинок, от класса — первые max_per_class.

        Классы идут в порядке хэша метки и берутся целиком, чтобы у запросов остались позитивы. Брать классы в порядке файла нельзя:
        первым в файле чаще встречается крупный класс, и выборка состояла бы из одних популярных вин.
        """
        rows_by_label: dict[str, list[int]] = {}
        for i, label in enumerate(self.labels):
            rows_by_label.setdefault(label, []).append(i)
        rows: list[int] = []
        for label in sorted(rows_by_label, key=lambda name: hashlib.blake2b(name.encode(), digest_size=8).digest()):
            if max_images is not None and len(rows) >= max_images:
                break
            rows.extend(rows_by_label[label][:max_per_class])
        return self.take(sorted(rows))

    def subsample(self, max_images: int | None) -> "Items":
        """Не больше max_images картинок равномерно по файлу; для ролей, где классы не важны."""
        if max_images is None or max_images >= len(self):
            return self
        return self.take([int(i) for i in np.linspace(0, len(self) - 1, max_images).round()])

    def class_sizes(self) -> Counter[str]:
        """Сколько картинок в каждом классе."""
        return Counter(self.labels)


def index_markup(root: Path, wanted: set[str]) -> dict[str, Span]:
    """Смещения строк normalization.jsonl для картинок из wanted, у которых нашлась годная бутылка.

    Полигоны в память не берутся: разметка WineSensed весит под гигабайт, воркеры даталоадера читают строку по смещению.
    Картинка из wanted без строки разметки — ошибка: датасет размечен не до конца.
    """
    path = root / MARKUP_NAME
    logger.info(f"Индексирую {path}: нужно {len(wanted)} картинок")
    spans: dict[str, Span] = {}
    left = set(wanted)
    offset = 0
    with path.open("rb") as f:
        for line in f:
            if not left:
                break
            rec = json.loads(line)
            if rec["file"] in left:
                left.discard(rec["file"])
                if rec["candidates"]:
                    spans[rec["file"]] = (offset, len(line))
            offset += len(line)
    if left:
        raise ValueError(f"в {path} нет разметки для {len(left)} картинок, например {sorted(left)[:3]}: доразметьте датасет через scripts.normalize_dataset")
    return spans


def read_items(root: Path, files: list[str]) -> list[Items]:
    """Сплиты датасета из json вида {имя файла: метка} с привязкой к разметке; normalization.jsonl читается один раз на все файлы."""
    labels_by_file = []
    for file in files:
        with (root / file).open(encoding="utf-8") as f:
            labels_by_file.append(json.load(f))
    spans = index_markup(root, {name for labels in labels_by_file for name in labels})
    return [Items(root, list(labels), list(labels.values()), [spans.get(name) for name in labels]) for labels in labels_by_file]


def read_candidate(root: Path, span: Span) -> dict:
    """Лучшая по скору отбора годная бутылка картинки: полигоны bottle и label, angle."""
    offset, length = span
    with (root / MARKUP_NAME).open("rb") as f:
        f.seek(offset)
        return json.loads(f.read(length))["candidates"][0]
