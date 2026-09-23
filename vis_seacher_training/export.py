import json
from pathlib import Path

import numpy as np
import rich_click as click
import torch
from kostyl.utils import setup_logger
from torch import nn
from torch.export import Dim

from vis_seacher_training.dino_modeling import DinoV3ForWine

logger = setup_logger(fmt="detailed")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OPSET = 23  # с 23 у ONNX есть оператор Attention, и sdpa уходит в него целиком, а не россыпью matmul и softmax
ONNX_NAME = "model.onnx"
DUMMY_BATCH = 2
"""Пример входа для трассировки: батч и обе стороны больше единицы и не равны друг другу, иначе экспортёр счёл бы их константами либо одной осью."""


class WineEncoderForExport(nn.Module):
    """Обёртка для инференса: на входе картинки float32 в диапазоне 0…255, RGB, NCHW; нормировка ImageNet вшита в граф, на выходе L2-нормированные эмбеддинги."""

    def __init__(
        self,
        model: DinoV3ForWine,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.model = model.eval()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.eval()  # иначе обёртка остаётся в режиме обучения, и экспортёр предупреждает, хотя модель внутри уже в eval

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixel_values = (images / 255.0 - self.mean) / self.std  # ty: ignore[unsupported-operator]
        return self.model(pixel_values, normalize=True)


def load_best(checkpoint: Path) -> DinoV3ForWine:
    """Модель из чекпоинта Lightning в float32 на CPU. Внимание переключается на sdpa: flash attention в ONNX не уходит, а веса от него не зависят."""
    model = DinoV3ForWine.from_lightning_checkpoint(
        checkpoint, attn_implementation="sdpa", dtype=torch.float32
    )
    return model.cpu().eval()


def dummy_images(patch: int, height: int, width: int) -> torch.Tensor:
    """Пример входа: батч DUMMY_BATCH, стороны из конфига данных, при равных сторонах ширина укорачивается на патч, чтобы оси остались разными."""
    if height == width:
        width -= patch
    return torch.rand(DUMMY_BATCH, 3, height, width) * 255


def export_onnx(
    encoder: WineEncoderForExport, path: Path, patch: int, input_size: tuple[int, int]
) -> torch.Tensor:
    """ONNX одним файлом с динамическими батчем и сторонами картинки; стороны — кратные патчу, это записано в осях как patch × число патчей."""
    images = dummy_images(patch, *input_size)
    dynamic_shapes = {
        "images": {
            0: Dim("batch"),
            2: patch * Dim("height_patches"),
            3: patch * Dim("width_patches"),
        }
    }
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            (images,),
            path,
            dynamo=True,
            external_data=False,
            opset_version=OPSET,
            input_names=["images"],
            output_names=["embeddings"],
            dynamic_shapes=dynamic_shapes,
        )
    logger.info(
        f"ONNX записан в {path}: {path.stat().st_size / 2**20:.0f} МБ, вход {tuple(images.shape)}, оси batch и стороны динамические"
    )
    return images


def verify_onnx(
    encoder: WineEncoderForExport, path: Path, images: torch.Tensor, patch: int
) -> float:
    """Сверка с PyTorch на трассировочном входе и на другом батче с другими сторонами: возвращает наибольшее расхождение по модулю."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    other = (
        torch.rand(DUMMY_BATCH + 1, 3, images.shape[2] + patch, images.shape[3] - patch)
        * 255
    )
    worst = 0.0
    with torch.no_grad():
        for batch in (images, other):
            want = encoder(batch).numpy()
            (got,) = session.run(["embeddings"], {"images": batch.numpy()})
            diff = float(np.abs(want - got).max())
            worst = max(worst, diff)
            logger.info(f"вход {tuple(batch.shape)}: расхождение с PyTorch {diff:.2e}")
    return worst


def export_model(
    model: DinoV3ForWine,
    output: Path,
    input_size: tuple[int, int],
    fill: tuple[int, int, int],
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> None:
    """Всё, что нужно инференсу, в одну директорию: веса save_pretrained, ONNX с вшитой нормировкой и описание входа."""
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    logger.info(f"Модель сохранена в {output}")
    encoder = WineEncoderForExport(model, mean, std)
    patch = model.config.patch_size
    images = export_onnx(encoder, output / ONNX_NAME, patch, input_size)
    worst = verify_onnx(encoder, output / ONNX_NAME, images, patch)
    if worst > 1e-3:
        raise RuntimeError(
            f"ONNX расходится с PyTorch на {worst:.2e}: экспорт не годится"
        )
    contract = {
        "input": "images: float32, RGB, NCHW, значения 0…255; нормировка ImageNet внутри графа",
        "output": "embeddings: float32 (batch, embed_dim), L2-нормированные",
        "input_size": list(input_size),
        "patch_size": patch,
        "pad_color": list(fill),
        "preprocess": "кроп нормализации вписать в input_size с сохранением пропорций (area при уменьшении, cubic при увеличении), дополнить pad_color по центру",
        "mean": list(mean),
        "std": list(std),
        "onnx_max_abs_diff_vs_torch": worst,
    }
    (output / "preprocess.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=1), encoding="utf-8"
    )


@click.command()
@click.option(
    "-c",
    "--checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Чекпоинт Lightning (.ckpt), обычно лучший из checkpoints/",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Директория модели: save_pretrained, model.onnx и preprocess.json",
)
@click.option(
    "--input-size",
    type=(int, int),
    default=(512, 512),
    show_default=True,
    help="Высота и ширина входа, как data.input_size обучения",
)
@click.option(
    "--fill",
    type=(int, int, int),
    default=(124, 116, 104),
    show_default=True,
    help="Цвет полей, как background.color в normalize.toml",
)
def main(
    checkpoint: Path,
    output: Path,
    input_size: tuple[int, int],
    fill: tuple[int, int, int],
) -> None:
    """Экспорт обученной модели из чекпоинта; run.py делает то же сам после обучения. Запуск из корня: python -m vis_seacher_training.export."""
    export_model(load_best(checkpoint), output, input_size, fill)


if __name__ == "__main__":
    main()
