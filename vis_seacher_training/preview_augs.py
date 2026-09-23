from pathlib import Path

import cv2
import numpy as np
import rich_click as click

from vinishko.pipeline.steps.normalization.normalize import HERE as NORM_DIR
from vinishko.pipeline.steps.normalization.normalize import load_config
from vis_seacher_training.configs import TrainingConfig
from vis_seacher_training.data.augmentations import TrainAugmenter
from vis_seacher_training.data.dataset import WineViewDataset
from vis_seacher_training.data.markup import read_items


def to_image(pixels: np.ndarray, mean: tuple[float, ...], std: tuple[float, ...]) -> np.ndarray:
    """Вход модели обратно в uint8; поля выходят цветом заливки фона, которым и были добавлены."""
    return ((pixels.transpose(1, 2, 0) * np.array(std) + np.array(mean)) * 255).clip(0, 255).round().astype(np.uint8)


@click.command()
@click.option("-c", "--config", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="config.yaml эксперимента: из него берутся данные, размер входа и аугментации")
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path), required=True, help="Куда сохранить png")
@click.option("--images", type=click.IntRange(min=1), default=8, show_default=True, help="Сколько фото показать, строка на фото")
@click.option("--views", type=click.IntRange(min=1), default=9, show_default=True, help="Сколько аугментированных видов на фото")
@click.option("--seed", type=int, default=0, show_default=True)
def main(config: Path, output: Path, images: int, views: int, seed: int) -> None:
    """Лист аугментаций: в каждой строке вход модели без аугментаций, как на валидации и в бою, и случайные виды той же бутылки. Запуск из корня: python -m vis_seacher_training.preview_augs."""
    cfg = TrainingConfig.from_file(config).data
    render_cfg = load_config(cfg.normalize_config or NORM_DIR / "normalize.toml", [])
    (items,) = read_items(cfg.datasets_dir / "winesensed", ["val.json"])
    items = items.with_bottle().sample_classes(2000)
    rows = np.random.default_rng(seed).choice(len(items), images, replace=False)
    common = (items, [0] * len(items), render_cfg, cfg.input_size, cfg.img_mean, cfg.img_std)
    clean = WineViewDataset(*common)
    augmented = WineViewDataset(*common, augmenter=TrainAugmenter(cfg.augmentations, render_cfg, cfg.input_size, seed))
    sheet = [np.concatenate([to_image(ds[int(row)]["pixel_values"].numpy(), cfg.img_mean, cfg.img_std) for ds in [clean] + [augmented] * views], axis=1) for row in rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(np.concatenate(sheet, axis=0), cv2.COLOR_RGB2BGR))
    click.echo(f"{output}: {images} фото × (1 без аугментаций + {views} видов)")


if __name__ == "__main__":
    main()
