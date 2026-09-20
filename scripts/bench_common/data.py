import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rich_click as click
from PIL import Image
from rich.progress import Progress
from torch.utils.data import Dataset

from vinishko.pipeline.steps.normalization.normalize import render_bottle
from vinishko.pipeline.steps.normalization.seg import open_image

MARKUP_NAME = "normalization.jsonl"

Span = tuple[int, int]
"""Где в normalization.jsonl лежит строка картинки: смещение и длина в байтах."""


@dataclass(frozen=True)
class Split:
    """Картинки одной роли замера: val — галерея и запросы, distractors и negatives — только запросы без позитивов."""

    role: str
    root: Path
    """Директория развёрнутого датасета: картинки в images, разметка нормализации в normalization.jsonl."""
    names: list[str]
    labels: list[str]
    spans: list[Span] | None = None
    """Строки разметки по картинкам; None — картинка идёт в модель целиком, без нормализации."""

    def __len__(self) -> int:
        return len(self.names)

    def path(self, i: int) -> Path:
        """Путь до картинки i."""
        return self.root / "images" / self.names[i]

    def part(self, start: int, stop: int) -> "Split":
        """Картинки с start по stop: кусок сплита для одного устройства."""
        return replace(self, names=self.names[start:stop], labels=self.labels[start:stop], spans=None if self.spans is None else self.spans[start:stop])


def read_split(root: Path, file: str, role: str) -> Split:
    """Сплит из json вида {имя файла: метка}, как его пишет prepare_datasets."""
    with (root / file).open(encoding="utf-8") as f:
        labels: dict[str, str] = json.load(f)
    return Split(role, root, list(labels), list(labels.values()))


def limit_split(split: Split, limit: int) -> Split:
    """Первые limit картинок сплита; классы берутся целиком, чтобы у запросов val остались позитивы в галерее."""
    rows_by_label: dict[str, list[int]] = {}
    for i, label in enumerate(split.labels):
        rows_by_label.setdefault(label, []).append(i)
    rows: list[int] = []
    for members in rows_by_label.values():
        if len(rows) >= limit:
            break
        rows.extend(members)
    return replace(split, names=[split.names[i] for i in rows], labels=[split.labels[i] for i in rows])


def index_markup(root: Path, wanted: set[str], progress: Progress) -> dict[str, Span]:
    """Смещения строк normalization.jsonl для картинок из wanted, у которых нормализация нашла годную бутылку.

    Сами полигоны в память не берутся: файл WineSensed весит под гигабайт, воркеры даталоадера читают нужную строку по смещению.
    Картинка из wanted без строки разметки — ошибка: датасет размечен не до конца.
    """
    path = root / MARKUP_NAME
    if not path.exists():
        raise click.ClickException(f"нет {path}: разметьте датасет через python -m scripts.normalize_dataset run {root}")
    task = progress.add_task(path.name, name=f"разметка {root.name}", total=path.stat().st_size)
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
            progress.update(task, completed=offset)
    progress.remove_task(task)
    if left:
        raise click.ClickException(f"в {path} нет разметки для {len(left)} картинок, например {sorted(left)[:3]}: доразметьте датасет через scripts.normalize_dataset")
    return spans


def normalized(split: Split, spans: dict[str, Span]) -> Split:
    """Сплит после нормализации: остаются картинки с годной бутылкой, остальные пайплайн до энкодера не допускает."""
    rows = [i for i, name in enumerate(split.names) if name in spans]
    return replace(
        split,
        names=[split.names[i] for i in rows],
        labels=[split.labels[i] for i in rows],
        spans=[spans[split.names[i]] for i in rows],
    )


def query_rows(split: Split) -> np.ndarray:
    """Строки val, которые идут запросами leave-one-out: у класса в галерее есть ещё хотя бы одна картинка."""
    sizes = Counter(split.labels)
    return np.array([i for i, label in enumerate(split.labels) if sizes[label] >= 2], dtype=np.int64)


def load_view(split: Split, i: int, render_cfg: dict) -> Image.Image:
    """Картинка в том виде, в каком её получает энкодер: целиком либо кроп нормализации по сохранённой разметке.

    Кроп рендерится той же функцией, что в пайплайне: первая, то есть лучшая по скору отбора, годная бутылка картинки.
    """
    img = open_image(split.path(i))
    if split.spans is None:
        return img
    offset, length = split.spans[i]
    with (split.root / MARKUP_NAME).open("rb") as f:
        f.seek(offset)
        cand = json.loads(f.read(length))["candidates"][0]
    crop, _ = render_bottle(np.asarray(img), cand["bottle"], cand["label"], render_cfg, angle=cand["angle"])
    return Image.fromarray(crop)


class Views(Dataset):
    """Даталоадеру: вход энкодера после preprocess и номер картинки в сплите."""

    def __init__(self, split: Split, preprocess: Callable[[Image.Image], Any], render_cfg: dict) -> None:
        self.split, self.preprocess, self.render_cfg = split, preprocess, render_cfg

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        return self.preprocess(load_view(self.split, index, self.render_cfg)), index


def init_worker(_: int) -> None:
    """Воркер даталоадера — один поток: иначе OpenCV в каждом из них поднимает пул на все ядра."""
    cv2.setNumThreads(0)
