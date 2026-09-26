import json
import time
from pathlib import Path

import numpy as np
import rich_click as click

from vinishko.pipeline.steps.normalization.normalize import HERE as NORM_DIR
from vinishko.pipeline.steps.normalization.normalize import load_config
from vinishko.pipeline.steps.normalization.seg import open_image
from vis_seacher_training.configs import TrainingConfig
from vis_seacher_training.data.augmentations import TrainAugmenter
from vis_seacher_training.data.markup import index_markup
from vis_seacher_training.data.markup import read_candidate


def render_all(config: Path, images: int, seeds: int) -> dict[str, np.ndarray]:
    """Виды первых images картинок val с бутылкой × seeds сидов; ключ — имя файла и seed."""
    cfg = TrainingConfig.from_file(config).data
    render_cfg = load_config(cfg.normalize_config or NORM_DIR / "normalize.toml", [])
    root = cfg.datasets_dir / "winesensed"
    with (root / "val.json").open(encoding="utf-8") as f:
        names = list(json.load(f))[
            : images * 40 : 40
        ]  # с шагом, чтобы не взять один класс
    spans = index_markup(root, set(names))
    out: dict[str, np.ndarray] = {}
    for name in [n for n in names if n in spans][:images]:
        rgb = np.asarray(open_image(root / "images" / name))
        cand = read_candidate(root, spans[name])
        for seed in range(seeds):
            image, offset = TrainAugmenter(
                cfg.augmentations, render_cfg, cfg.input_size, seed
            )(rgb, cand)
            out[f"{name}|{seed}"], out[f"{name}|{seed}|offset"] = (
                image,
                np.array(offset),
            )
    return out


@click.command()
@click.argument("mode", type=click.Choice(["save", "check"]))
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="config.yaml эксперимента",
)
@click.option(
    "-o",
    "--fixture",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="npz с эталонными видами",
)
@click.option("--images", type=click.IntRange(min=1), default=12, show_default=True)
@click.option("--seeds", type=click.IntRange(min=1), default=8, show_default=True)
def main(mode: str, config: Path, fixture: Path, images: int, seeds: int) -> None:
    """Эталон аугментаций для правок без изменения поведения: save — записать виды, check — сравнить с ними бит в бит. Запуск из корня: python -m vis_seacher_training.augs_fixture."""
    started = time.time()
    views = render_all(config, images, seeds)
    click.echo(f"{len(views) // 2} видов за {time.time() - started:.1f} с")
    if mode == "save":
        fixture.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fixture, **views)  # ty: ignore[invalid-argument-type]
        click.echo(f"эталон записан в {fixture}")
        return
    ref = np.load(fixture)
    bad = 0
    for key, got in views.items():
        want = ref[key]
        if want.shape != got.shape:
            bad += 1
            click.echo(f"{key}: форма {want.shape} → {got.shape}")
        elif not np.array_equal(want, got):
            bad += 1
            diff = np.abs(want.astype(int) - got.astype(int))
            click.echo(
                f"{key}: отличаются {int((diff > 0).sum())} значений, максимум на {int(diff.max())}"
            )
    if bad:
        raise SystemExit(f"расхождений: {bad} из {len(views)}")
    click.echo("всё совпало")


if __name__ == "__main__":
    main()
