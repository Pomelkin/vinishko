import os
from pathlib import Path
from typing import override

import lightning as L
import torch
from kostyl.utils import dump_to_file
from kostyl.utils import setup_logger
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torch.utils.data import get_worker_info

from vinishko.pred.pipeline.steps.normalization.normalize import HERE as NORM_DIR
from vinishko.pred.pipeline.steps.normalization.normalize import load_config
from vis_seacher_training.configs import DataConfig
from vis_seacher_training.data.augmentations import TrainAugmenter
from vis_seacher_training.data.dataset import WineViewDataset
from vis_seacher_training.data.markup import Items
from vis_seacher_training.data.markup import read_items


logger = setup_logger(fmt="detailed")

VAL_LOADERS = ("val", "distractors", "negatives", "catalog", "catalog_queries")
"""Порядок даталоадеров валидации; по dataloader_idx модуль обучения понимает, чьи это эмбеддинги."""


def init_worker(_: int) -> None:
    """Свой поток случайности аугментаций в каждом воркере каждого ранга.

    torch даёт воркерам разные сиды, но у всех рангов DDP они совпадают, поэтому подмешивается ранг. Без пересидирования
    A.Compose после fork выдавал бы во всех воркерах одни и те же аугментации: у него собственный генератор.
    """
    info = get_worker_info()
    if info is not None and isinstance(info.dataset, WineViewDataset):
        info.dataset.reseed(
            (info.seed + 1000003 * int(os.environ.get("RANK", "0"))) % (1 << 32)
        )


class WineDataModule(L.LightningDataModule):
    """Данные обучения визуального энкодера.

    Обучение — winesensed/train.json. Валидация — пять даталоадеров, см. VAL_LOADERS: val и distractors из WineSensed, negatives из
    Products-10K, catalog из OFF и catalog_queries — аугментированные виды тех же фото OFF, у каждой картинки одинаковые от эпохи к эпохе.
    Везде, кроме negatives, берутся только картинки с годной бутылкой: остальные нормализатор до энкодера не допускает.
    Не-вино Products-10K без бутылки идёт в модель целиком: это худший случай, в бою его отсеял бы нормализатор.
    """

    def __init__(self, config: DataConfig, workdir: Path | None = None) -> None:
        super().__init__()
        self.config = config
        self.render_cfg = load_config(
            config.normalize_config or NORM_DIR / "normalize.toml", []
        )
        root = config.datasets_dir
        train, val, distractors = read_items(
            root / "winesensed", ["train.json", "val.json", "val_distractors.json"]
        )
        (catalog,) = read_items(root / "off", ["catalog.json"])
        (negatives,) = read_items(root / "products10k", ["negatives.json"])

        self.train_items = train.with_bottle().sample_classes(config.max_train_images)
        self.class_names = sorted(set(self.train_items.labels))
        sizes = self.train_items.class_sizes()
        self.class_counts = torch.tensor([sizes[name] for name in self.class_names])

        ev = config.eval
        self.val_items: dict[str, Items] = {
            "val": val.with_bottle().sample_classes(
                ev.max_val_images, ev.max_images_per_class
            ),
            "distractors": distractors.with_bottle().subsample(ev.max_distractors),
            "negatives": negatives.subsample(ev.max_negatives),
            "catalog": catalog.with_bottle(),
        }
        self.val_items["catalog_queries"] = self.val_items["catalog"]
        val_ids = {
            name: n for n, name in enumerate(sorted(set(self.val_items["val"].labels)))
        }
        self.val_labels = torch.tensor(
            [val_ids[label] for label in self.val_items["val"].labels]
        )

        self.train_dataset: WineViewDataset | None = None
        self.val_datasets: list[WineViewDataset] | None = None
        logger.log_rank_zero(
            "INFO",
            f"train: {len(self.train_items)} картинок, {len(self.class_names)} классов; "
            + ", ".join(
                f"{name}: {len(items)}" for name, items in self.val_items.items()
            ),
        )
        if workdir is not None:
            workdir.mkdir(parents=True, exist_ok=True)
            dump_to_file(
                {name: n for n, name in enumerate(self.class_names)},
                workdir / "label2id.json",
            )

    @property
    def num_classes(self) -> int:
        """Число классов обучения."""
        return len(self.class_names)

    def _dataset(
        self,
        items: Items,
        class_ids: list[int],
        augmenter: TrainAugmenter | None = None,
        deterministic_seed: int | None = None,
    ) -> WineViewDataset:
        cfg = self.config
        return WineViewDataset(
            items,
            class_ids,
            self.render_cfg,
            cfg.input_size,
            cfg.img_mean,
            cfg.img_std,
            augmenter,
            deterministic_seed,
        )

    def _augmenter(self) -> TrainAugmenter:
        return TrainAugmenter(
            self.config.augmentations, self.render_cfg, self.config.input_size
        )

    @override
    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return  # estimate_total_steps зовёт setup повторно
        label2id = {name: n for n, name in enumerate(self.class_names)}
        augmenter = self._augmenter() if self.config.augmentations.enabled else None
        self.train_dataset = self._dataset(
            self.train_items,
            [label2id[label] for label in self.train_items.labels],
            augmenter,
        )
        self.val_datasets = []
        for name in VAL_LOADERS:
            items = self.val_items[name]
            class_ids = self.val_labels.tolist() if name == "val" else [-1] * len(items)
            if name == "catalog_queries":
                self.val_datasets.append(
                    self._dataset(
                        items,
                        class_ids,
                        self._augmenter(),
                        self.config.eval.off_augment_seed,
                    )
                )
            else:
                self.val_datasets.append(self._dataset(items, class_ids))

    @override
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.train_dataset is None:
            raise RuntimeError("setup не вызывался")
        cfg = self.config
        return DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,  # BatchNorm головы не переносит батч из одной картинки
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
            worker_init_fn=init_worker,
        )

    @override
    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.val_datasets is None:
            raise RuntimeError("setup не вызывался")
        cfg = self.config
        return [
            DataLoader(
                dataset,
                batch_size=cfg.eval_batch_size or cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
                worker_init_fn=init_worker,
            )
            for dataset in self.val_datasets
        ]
