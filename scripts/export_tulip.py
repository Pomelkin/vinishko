"""Экспорт визуальной башни TULIP в формат моделей визуального поиска: model.onnx, model.bf16.onnx, config.json, preprocess.json.

Тот же контракт, что у экспорта DinoV3ForWine: вход float32 RGB NCHW 0…255, нормировка внутри графа, выход L2-нормированные эмбеддинги.
Директорию результата можно выложить на Hugging Face и подставить в config.yaml поиска вместо DINO. Запуск из корня:
python -m scripts.export_tulip -o weights/tulip-so400m-14-384-export
"""

import copy
import json
import tempfile
import re
from pathlib import Path

import numpy as np
import onnx
import open_clip
import rich_click as click
import torch
import torch.nn.functional as F
from kostyl.utils import setup_logger
from torch import nn
from torch.export import Dim

from vis_seacher_training.export import BF16_MIN_COSINE, DUMMY_BATCH, ONNX_BF16_NAME, ONNX_NAME, OPSET

logger = setup_logger(fmt="detailed")

MODEL = "TULIP-so400m-14-384"
PRETRAINED_TAG = "webli"
"""Из записи этого тега форк open_clip берёт предобработку TULIP: mean и std 0.5, bicubic, squash."""
DEFAULT_CHECKPOINT = Path("weights/tulip-so400m-14-384.ckpt")
RESIZES = ("squash", "pad")


class TulipEncoderForExport(nn.Module):
    """Визуальная башня TULIP под контракт поиска: вход 0…255, нормировка в графе, на выходе L2-нормированные эмбеддинги float32.

    compute_dtype — тип весов и вычислений: float32 для onnxruntime, bfloat16 для TensorRT.
    """

    def __init__(self, visual: nn.Module, mean: tuple[float, ...], std: tuple[float, ...], compute_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.visual = visual.to(compute_dtype).eval()
        self.compute_dtype = compute_dtype
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixel_values = (images / 255.0 - self.mean) / self.std  # ty: ignore[unsupported-operator]
        return F.normalize(self.visual(pixel_values.to(self.compute_dtype)).float(), dim=-1)


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
    visual.load_state_dict({k.removeprefix("visual."): v for k, v in state.items() if k.startswith("visual.")}, strict=True)
    return visual.float().eval().requires_grad_(False)


def export_onnx(encoder: TulipEncoderForExport, path: Path, input_size: tuple[int, int]) -> torch.Tensor:
    """ONNX одним файлом; динамический только батч: у TULIP позиционные эмбеддинги под фиксированный вход."""
    images = torch.rand(DUMMY_BATCH, 3, *input_size) * 255
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
            dynamic_shapes={"images": {0: Dim("batch")}},
        )
    inline_external_data(path)
    logger.info(f"ONNX записан в {path}: {path.stat().st_size / 2**20:.0f} МБ, вход {tuple(images.shape)}, ось batch динамическая")
    return images


def inline_external_data(path: Path) -> None:
    """Граф одним файлом: экспортёр выносит веса в <имя>.data, когда считает граф большим, а контракт поиска ждёт один файл до 2 ГБ."""
    data = path.with_name(path.name + ".data")
    if not data.is_file():
        return
    model = onnx.load(str(path), load_external_data=True)
    onnx.save_model(model, str(path))
    data.unlink()
    logger.info(f"веса из {data.name} вложены в {path.name}")


def verify_onnx(encoder: TulipEncoderForExport, path: Path, images: torch.Tensor) -> float:
    """Сверка с PyTorch на трассировочном входе и на батче другого размера: наибольшее расхождение по модулю."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    other = torch.rand(DUMMY_BATCH + 1, 3, *images.shape[2:]) * 255
    worst = 0.0
    with torch.no_grad():
        for batch in (images, other):
            want = encoder(batch).numpy()
            (got,) = session.run(["embeddings"], {"images": batch.numpy()})
            diff = float(np.abs(want - got).max())
            worst = max(worst, diff)
            logger.info(f"вход {tuple(batch.shape)}: расхождение с PyTorch {diff:.2e}")
    return worst


def verify_tensorrt(reference: TulipEncoderForExport, path: Path, images: torch.Tensor, input_size: tuple[int, int]) -> float | None:
    """Сверка bf16-графа через TensorRT с PyTorch float32: наименьший косинус по батчу; None — TensorRT не установлен."""
    from vinishko.pipeline.steps.vis_searcher.backends import TensorRTRunner, tensorrt_available

    if not tensorrt_available():
        logger.warning(f"{path.name} не сверен: нет tensorrt либо CUDA")
        return None
    with tempfile.TemporaryDirectory(prefix="trt-verify-") as tmp:  # engine сверки одноразовый: не в директорию модели, которая уезжает на HF, и не в общий кэш, где имена совпадают
        runner = TensorRTRunner(path, torch.device("cuda", 0), max_batch=images.shape[0], input_size=input_size, cache_dir=Path(tmp))
        with torch.no_grad():
            want = reference(images).numpy()
        got = runner(images)
        cosine = float((want * got).sum(1).min())
        logger.info(f"{path.name} через TensorRT: наименьший косинус с PyTorch float32 {cosine:.5f}")
    return cosine


@click.command()
@click.option("-o", "--output", type=click.Path(file_okay=False, path_type=Path), required=True, help="Директория результата: model.onnx, model.bf16.onnx, config.json, preprocess.json")
@click.option("--checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=DEFAULT_CHECKPOINT, show_default=True, help="Чекпоинт TULIP форка open_clip")
@click.option("--model", default=MODEL, show_default=True, help="Имя модели в форке open_clip")
@click.option("--resize", type=click.Choice(RESIZES), default="squash", show_default=True, help="Как поиск готовит вход: squash — растянуть до размера входа, как в замерах TULIP; pad — вписать с полями, как у DINO")
@click.option("--fill", type=(int, int, int), default=(124, 116, 104), show_default=True, help="Цвет полей для resize=pad, как background.color в normalize.toml")
@click.option("--bf16/--no-bf16", default=True, show_default=True, help="Вдобавок к float32-графу писать model.bf16.onnx для TensorRT")
def main(output: Path, checkpoint: Path, model: str, resize: str, fill: tuple[int, int, int], bf16: bool) -> None:
    """Экспорт визуальной башни TULIP под контракт визуального поиска; результат выкладывается на Hugging Face и подставляется в config.yaml."""
    output.mkdir(parents=True, exist_ok=True)
    visual = load_visual(model, checkpoint)
    cfg = open_clip.get_pretrained_cfg(model, PRETRAINED_TAG)
    mean, std = tuple(float(v) for v in cfg["mean"]), tuple(float(v) for v in cfg["std"])
    model_cfg = open_clip.get_model_config(model)
    if model_cfg is None:
        raise RuntimeError(f"в форке open_clip нет конфига модели {model}")
    vision_cfg = model_cfg["vision_cfg"]
    size = vision_cfg["image_size"]
    input_size = (int(size[0]), int(size[1])) if isinstance(size, list | tuple) else (int(size), int(size))
    patch = patch_size_of(vision_cfg)
    encoder = TulipEncoderForExport(copy.deepcopy(visual), mean, std)
    with torch.no_grad():
        embed_dim = int(encoder(torch.zeros(1, 3, *input_size)).shape[1])
    logger.info(f"{model}: вход {input_size}, патч {patch}, эмбеддинг {embed_dim}, mean {mean}, std {std}")
    images = export_onnx(encoder, output / ONNX_NAME, input_size)
    worst = verify_onnx(encoder, output / ONNX_NAME, images)
    if worst > 1e-3:
        raise RuntimeError(f"ONNX расходится с PyTorch на {worst:.2e}: экспорт не годится")
    trt_cosine = None
    if bf16:
        half = TulipEncoderForExport(copy.deepcopy(visual), mean, std, torch.bfloat16)
        export_onnx(half, output / ONNX_BF16_NAME, input_size)
        trt_cosine = verify_tensorrt(encoder, output / ONNX_BF16_NAME, images, input_size)
        if trt_cosine is not None and trt_cosine < BF16_MIN_COSINE:
            raise RuntimeError(f"bf16-граф через TensorRT расходится с PyTorch: косинус {trt_cosine:.4f} < {BF16_MIN_COSINE}")
    (output / "config.json").write_text(
        json.dumps(
            {
                "model_type": "tulip_visual",
                "architecture": model,
                "open_clip_pretrained_tag": PRETRAINED_TAG,
                "source_checkpoint": checkpoint.name,
                "embed_dim": embed_dim,
                "image_size": list(input_size),
                "patch_size": patch,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    contract = {
        "input": "images: float32, RGB, NCHW, значения 0…255; нормировка внутри графа",
        "output": "embeddings: float32 (batch, embed_dim), L2-нормированные",
        "input_size": list(input_size),
        "patch_size": patch,
        "pad_color": list(fill),
        "resize": resize,
        "preprocess": "растянуть кроп до input_size без полей" if resize == "squash" else "вписать кроп в input_size с сохранением пропорций, дополнить pad_color по центру",
        "mean": list(mean),
        "std": list(std),
        "onnx_max_abs_diff_vs_torch": worst,
        "onnx_bf16": ONNX_BF16_NAME if bf16 else None,
        "onnx_bf16_min_cosine_vs_torch_tensorrt": trt_cosine,
    }
    (output / "preprocess.json").write_text(json.dumps(contract, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info(f"готово: {output}")


if __name__ == "__main__":
    main()
