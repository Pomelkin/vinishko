"""Экспорт визуальной башни TULIP в формат моделей визуального поиска: model.onnx, model.bf16.onnx, config.json, preprocess.json.

Директорию результата можно выложить на Hugging Face и подставить в config.yaml поиска вместо DINO. Запуск из корня:
python -m scripts.export_tulip -o weights/tulip_so400m_14_384
"""

import re
from pathlib import Path

import open_clip
import rich_click as click
import torch
from torch import nn

from scripts.export_common import RESIZES, Contract, export_bundle

MODEL = "TULIP-so400m-14-384"
PRETRAINED_TAG = "webli"
"""Из записи этого тега форк open_clip берёт предобработку TULIP: mean и std 0.5, bicubic, squash."""
DEFAULT_CHECKPOINT = Path("weights/tulip-so400m-14-384.ckpt")


def patch_size_of(vision_cfg: dict) -> int:
    """Патч башни: из конфига open_clip либо из имени модели timm вида vit_so400m_patch14_siglip_384, как у TULIP."""
    if "patch_size" in vision_cfg:
        return int(vision_cfg["patch_size"])
    match = re.search(r"patch(\d+)", str(vision_cfg.get("timm_model_name", "")))
    if match is None:
        raise RuntimeError(f"не понять патч башни из vision_cfg {vision_cfg}")
    return int(match.group(1))


def load_visual(model: str, checkpoint: Path) -> nn.Module:
    """Визуальная башня без базовых весов SigLIP с HF: чекпоинт содержит все ключи, грузятся ключи visual.* строго."""
    visual = open_clip.create_model(model, pretrained=None).visual
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    visual.load_state_dict(
        {
            k.removeprefix("visual."): v
            for k, v in state.items()
            if k.startswith("visual.")
        },
        strict=True,
    )
    return visual.float().eval().requires_grad_(False)


@click.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Директория результата: model.onnx, model.bf16.onnx, config.json, preprocess.json",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_CHECKPOINT,
    show_default=True,
    help="Чекпоинт TULIP форка open_clip",
)
@click.option(
    "--model", default=MODEL, show_default=True, help="Имя модели в форке open_clip"
)
@click.option(
    "--resize",
    type=click.Choice(RESIZES),
    default="squash",
    show_default=True,
    help="Как поиск готовит вход: squash — растянуть до размера входа, как в замерах TULIP; pad — вписать с полями, как у DINO",
)
@click.option(
    "--fill",
    type=(int, int, int),
    default=(124, 116, 104),
    show_default=True,
    help="Цвет полей для resize=pad, как background.color в normalize.toml",
)
@click.option(
    "--bf16/--no-bf16",
    default=True,
    show_default=True,
    help="Вдобавок к float32-графу писать model.bf16.onnx для TensorRT",
)
def main(
    output: Path,
    checkpoint: Path,
    model: str,
    resize: str,
    fill: tuple[int, int, int],
    bf16: bool,
) -> None:
    """Экспорт визуальной башни TULIP под контракт визуального поиска; результат выкладывается на Hugging Face и подставляется в config.yaml."""
    visual = load_visual(model, checkpoint)
    cfg = open_clip.get_pretrained_cfg(model, PRETRAINED_TAG)
    model_cfg = open_clip.get_model_config(model)
    if model_cfg is None:
        raise RuntimeError(f"в форке open_clip нет конфига модели {model}")
    vision_cfg = model_cfg["vision_cfg"]
    size = vision_cfg["image_size"]
    contract = Contract(
        input_size=(int(size[0]), int(size[1]))
        if isinstance(size, list | tuple)
        else (int(size), int(size)),
        patch_size=patch_size_of(vision_cfg),
        mean=tuple(float(v) for v in cfg["mean"]),
        std=tuple(float(v) for v in cfg["std"]),
        resize=resize,
        pad_color=fill,
        interpolation=str(cfg["interpolation"]),  # bicubic: так делает трансформ форка
    )
    config = {
        "model_type": "tulip_visual",
        "architecture": model,
        "open_clip_pretrained_tag": PRETRAINED_TAG,
        "source_checkpoint": checkpoint.name,
    }
    export_bundle(visual, output, contract, config, bf16)


if __name__ == "__main__":
    main()
