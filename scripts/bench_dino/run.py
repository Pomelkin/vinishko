from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rich_click as click
import torch

from scripts.bench_common.cli import MODES
from scripts.bench_common.cli import PROTOCOLS
from scripts.bench_common.cli import common_options
from scripts.bench_common.cli import console
from scripts.bench_common.cli import count_progress
from scripts.bench_common.cli import dataset_paths
from scripts.bench_common.cli import finish
from scripts.bench_common.cli import prepare_mode
from scripts.bench_common.cli import read_splits
from scripts.bench_common.data import Split
from scripts.bench_common.data import protocol_rows
from scripts.bench_common.metrics import ModeResult
from scripts.bench_common.metrics import make_result
from scripts.bench_common.parallel import DevicePool
from scripts.bench_common.parallel import embed_parts
from scripts.bench_common.report import ReportSpec
from scripts.bench_dino.model import BACKENDS
from scripts.bench_dino.model import MODEL
from scripts.bench_dino.model import DinoWorker
from scripts.bench_dino.model import read_contract
from scripts.bench_tulip.search import (
    retrieve,
)  # точный faiss по косинусу: у обеих моделей один вектор на картинку
from vinishko.pipeline.steps.normalization.normalize import load_config

SPEC = ReportSpec(model=MODEL, score_name="косинус", score_short="cos")


def run_mode(
    mode: str, protocols: list[str], raw: dict[str, Split], pool: DevicePool, tmp: Path
) -> list[ModeResult]:
    """Замер одного режима: эмбеддинги всех ролей на устройствах пула, затем по каждому протоколу поиск по галерее и метрики."""
    splits, dropped = prepare_mode(mode, raw)
    vectors: dict[str, np.ndarray] = {}
    seconds = 0.0
    with count_progress() as progress:
        for role, split in splits.items():
            paths, _, elapsed = embed_parts(
                pool, split, tmp / f"{mode}_{role}", progress
            )
            vectors[role] = np.concatenate([np.load(path) for path in paths])
            seconds += elapsed
        results = []
        for protocol in protocols:
            gallery_rows, rows = protocol_rows(splits["val"], protocol)
            val = retrieve(
                splits["val"],
                vectors["val"],
                rows,
                vectors["val"],
                gallery_rows,
                True,
                progress,
            )
            rejects = {
                role: retrieve(
                    splits[role],
                    vectors[role],
                    np.arange(len(splits[role])),
                    vectors["val"],
                    gallery_rows,
                    False,
                    progress,
                )
                for role in splits
                if role != "val"
            }
            results.append(
                make_result(
                    mode,
                    protocol,
                    raw,
                    splits,
                    dropped,
                    len(gallery_rows),
                    val,
                    rejects,
                    seconds,
                )
            )
    return results


@click.command()
@common_options
@click.option(
    "--onnx",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Граф из экспорта обучения: model.onnx (float32) для onnxruntime, model.bf16.onnx для TensorRT; вход float32 0…255 RGB NCHW, нормировка внутри графа, выход L2-нормированные эмбеддинги",
)
@click.option(
    "--backend",
    type=click.Choice(BACKENDS),
    default="onnxruntime",
    show_default=True,
    help="Чем исполнять граф: onnxruntime с провайдером CUDA либо CPU, или TensorRT — нужен пакет tensorrt из группы flash-inference; engine собирается при первом запуске несколько минут и кэшируется рядом с ONNX",
)
@click.option(
    "--input-size",
    type=(int, int),
    default=None,
    help="Высота и ширина входа; по умолчанию из preprocess.json рядом с весами, иначе 512 512",
)
@click.option("--batch-size", type=click.IntRange(min=1), default=32, show_default=True)
def main(
    output: Path,
    devices: list[torch.device],
    mode: str,
    protocol: str,
    datasets_dir: Path,
    winesensed: Path | None,
    negatives: Path | None,
    distractors: bool,
    norm_config: Path,
    workers: int,
    limit: int | None,
    examples: int,
    onnx: Path,
    backend: str,
    input_size: tuple[int, int] | None,
    batch_size: int,
) -> None:
    """Замер обученной DinoV3ForWine из ONNX на WineSensed: recall@1/3/5 и отделимость запросов без ответа. Запуск из корня: python -m scripts.bench_dino.run.

    Протоколы loo и oneshot, режимы raw и norm, запросы без ответа и отчёт — те же, что у scripts.bench_tulip.run и scripts.bench_evie.run,
    поэтому отчёты сравнимы столбец в столбец. Вход модели готовится как в пайплайне и в обучении: кроп нормализации, а в режиме raw целое
    фото, вписывается в размер входа с сохранением пропорций и дополняется полями цвета заливки фона; нормировка ImageNet вшита в граф.
    Граф считается через onnxruntime (на CUDA нужна сборка onnxruntime-gpu) либо TensorRT, см. --backend; точность вычислений задана самим ONNX:
    model.onnx — float32, model.bf16.onnx — bfloat16, его исполняет только TensorRT. Поиск точный: faiss IndexFlatIP по L2-нормированным векторам.

    Несколько видеокарт в --device — параллелизм по данным: на каждой своя сессия onnxruntime, картинки сплита делятся между ними поровну.
    """
    render_cfg = load_config(norm_config, [])
    size, fill = read_contract(
        onnx, input_size, tuple(render_cfg["background"]["color"])
    )  # type: ignore[arg-type]
    winesensed, negatives = dataset_paths(datasets_dir, winesensed, negatives)
    raw = read_splits(winesensed, negatives, distractors, limit)
    with TemporaryDirectory(prefix="bench_dino_") as tmp:
        with console.status(f"загрузка {onnx.name} на {len(devices)} устр."):
            pool = DevicePool(
                devices,
                DinoWorker,
                (onnx, backend, size, fill, render_cfg, batch_size, workers),
            )
        with pool:
            console.print(
                f"[green]model[/] {MODEL} из {onnx} на {', '.join(pool.names)}, {pool.info[0]['provider']}"
            )
            results = [
                r
                for m in MODES[mode]
                for r in run_mode(m, PROTOCOLS[protocol], raw, pool, Path(tmp))
            ]
            info = pool.info[0]
    run_info = {
        "модель": f"{MODEL}, ONNX {onnx}, {onnx.stat().st_size / 2**20:.0f} МБ",
        "устройства": f"{', '.join(map(str, devices))}, {info['provider']}",
        "вход": f"{info['вход']}; кроп вписан с сохранением пропорций, нормировка ImageNet в графе",
        "поиск": "faiss IndexFlatIP, точный, по L2-нормированным векторам",
        "галерея и запросы": str(winesensed),
        "негативы": str(negatives),
        "конфиг нормализации": f"{norm_config}: crop={render_cfg['crop']}, background={render_cfg['background']}",
        "ограничение": "нет" if limit is None else f"--limit {limit}",
        "батч и воркеры на устройство": f"{batch_size}, {workers}",
    }
    finish(results, SPEC, run_info, render_cfg, examples, output)


if __name__ == "__main__":
    main()
