from typing import TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from vinishko.pipeline.steps.normalization.normalize import render_bottle
from vinishko.pipeline.steps.normalization.seg import open_image
from vis_seacher_training.data.augmentations import TrainAugmenter
from vis_seacher_training.data.augmentations import resize_to_fit
from vis_seacher_training.data.markup import Items
from vis_seacher_training.data.markup import read_candidate



class Sample(TypedDict):
    """Один пример даталоадера."""

    pixel_values: torch.Tensor
    label: int
    index: int



def to_tensor(image: np.ndarray, input_size: tuple[int, int], mean: tuple[float, ...], std: tuple[float, ...], offset: tuple[float, float]) -> torch.Tensor:
    """Нормировка и поля до размера входа. Поля добавляются уже после нормировки и потому равны нулю точно, а не с ошибкой округления цвета."""
    height, width = image.shape[:2]
    canvas = torch.zeros(3, *input_size)
    top, left = round((input_size[0] - height) * offset[0]), round((input_size[1] - width) * offset[1])
    pixels = (torch.from_numpy(image).float() / 255 - torch.tensor(mean)) / torch.tensor(std)
    canvas[:, top : top + height, left : left + width] = pixels.permute(2, 0, 1)
    return canvas


class WineViewDataset(Dataset):
    """Картинка → вход модели и номер класса.

    Без аугментера — ровно вход пайплайна: render_bottle по сохранённой разметке, ресайз с сохранением пропорций, картинка по центру полей.
    Картинка без годной бутылки идёт целиком: такие бывают только среди негативов, в бою их отсеял бы нормализатор.
    С аугментером — случайный вид той же бутылки; deterministic_seed делает его одинаковым от эпохи к эпохе, для синтетических запросов валидации.
    """

    def __init__(
        self,
        items: Items,
        class_ids: list[int],
        render_cfg: dict,
        input_size: tuple[int, int],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        augmenter: TrainAugmenter | None = None,
        deterministic_seed: int | None = None,
    ) -> None:
        self.items, self.class_ids, self.render_cfg = items, class_ids, render_cfg
        self.input_size, self.mean, self.std = input_size, mean, std
        self.augmenter, self.deterministic_seed = augmenter, deterministic_seed

    def __len__(self) -> int:
        return len(self.items)

    def reseed(self, seed: int) -> None:
        """Свой поток случайности аугментаций; зовётся в воркере даталоадера."""
        if self.augmenter is not None:
            self.augmenter.reseed(seed)

    def view(self, index: int) -> tuple[np.ndarray, tuple[float, float]]:
        """Кроп uint8, вписанный в размер входа, и его положение внутри полей."""
        rgb = np.asarray(open_image(self.items.path(index)))
        span = self.items.spans[index]
        if span is None:
            return resize_to_fit(rgb, self.input_size), (0.5, 0.5)
        cand = read_candidate(self.items.root, span)
        if self.augmenter is None:
            crop, _ = render_bottle(rgb, cand["bottle"], cand["label"], self.render_cfg, angle=cand["angle"])
            return resize_to_fit(crop, self.input_size), (0.5, 0.5)
        if self.deterministic_seed is not None:
            self.augmenter.reseed(self.deterministic_seed + index)
        return self.augmenter(rgb, cand)

    def __getitem__(self, index: int) -> Sample:
        image, offset = self.view(index)
        return {"pixel_values": to_tensor(image, self.input_size, self.mean, self.std, offset), "label": self.class_ids[index], "index": index}
