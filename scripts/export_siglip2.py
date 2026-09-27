"""Экспорт визуальной башни SigLIP2 с Hugging Face в формат моделей визуального поиска: model.onnx, model.bf16.onnx, config.json, preprocess.json.

Эмбеддинг картинки — pooler_output башни, то есть выход MAP-головы, как в get_image_features. Запуск из корня:
python -m scripts.export_siglip2 -o weights/siglip2_so400m_14_384
"""

import json
from pathlib import Path

import rich_click as click
import torch
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import SiglipVisionModel

from scripts.export_common import RESIZES, Contract, export_bundle

DEFAULT_REPO = "google/siglip2-so400m-patch14-384"
RESAMPLING = {2: "bilinear", 3: "bicubic"}
"""Коды resample препроцессора transformers (PIL): SigLIP масштабирует bilinear."""


class SiglipImageEmbedder(nn.Module):
    """Башня SigLIP → эмбеддинг картинки: pooler_output, выход MAP-головы."""

    def __init__(self, vision: SiglipVisionModel) -> None:
        super().__init__()
        self.vision = vision

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision(pixel_values=pixel_values).pooler_output


@click.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Директория результата: model.onnx, model.bf16.onnx, config.json, preprocess.json",
)
@click.option(
    "--repo",
    default=DEFAULT_REPO,
    show_default=True,
    help="Репозиторий SigLIP2 на Hugging Face с фиксированным входом",
)
@click.option("--revision", default=None, help="Ревизия репозитория; без неё последняя")
@click.option(
    "--resize",
    type=click.Choice(RESIZES),
    default="squash",
    show_default=True,
    help="Как поиск готовит вход: squash — растянуть до размера входа, как препроцессор SigLIP; pad — вписать с полями, как у DINO",
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
    repo: str,
    revision: str | None,
    resize: str,
    fill: tuple[int, int, int],
    bf16: bool,
) -> None:
    """Экспорт визуальной башни SigLIP2 под контракт визуального поиска; результат выкладывается на Hugging Face и подставляется в config.yaml."""
    vision = (
        SiglipVisionModel.from_pretrained(
            repo,
            revision=revision or "main",
            attn_implementation="sdpa",
            dtype=torch.float32,
        )
        .eval()
        .requires_grad_(False)
    )
    preprocessor = json.loads(
        Path(
            hf_hub_download(repo, "preprocessor_config.json", revision=revision)
        ).read_text(encoding="utf-8")
    )
    size = preprocessor["size"]
    vc = vision.config
    contract = Contract(
        input_size=(int(size["height"]), int(size["width"])),
        patch_size=int(vc.patch_size),
        mean=tuple(float(v) for v in preprocessor["image_mean"]),
        std=tuple(float(v) for v in preprocessor["image_std"]),
        resize=resize,
        pad_color=fill,
        interpolation=RESAMPLING[int(preprocessor["resample"])],
    )
    if contract.input_size != (int(vc.image_size), int(vc.image_size)):
        raise RuntimeError(
            f"препроцессор {contract.input_size} и башня {vc.image_size} расходятся по размеру входа"
        )
    config = {
        "model_type": "siglip2_vision",
        "architecture": repo,
        "hf_revision": revision or "main",
        "pooling": "map_head",
    }
    export_bundle(SiglipImageEmbedder(vision), output, contract, config, bf16)


if __name__ == "__main__":
    main()
