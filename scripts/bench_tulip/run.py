from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rich_click as click
import torch

from scripts.bench_common.cli import MODES
from scripts.bench_common.cli import PROTOCOLS
from scripts.bench_common.cli import ROOT
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
from scripts.bench_tulip.model import MODEL
from scripts.bench_tulip.model import RESIZE_MODES
from scripts.bench_tulip.model import TulipWorker
from scripts.bench_tulip.search import retrieve
from vinishko.pipeline.steps.normalization.normalize import load_config

SPEC = ReportSpec(model=MODEL, score_name="косинус", score_short="cos")


def run_mode(mode: str, protocols: list[str], raw: dict[str, Split], pool: DevicePool, tmp: Path) -> list[ModeResult]:
    """Замер одного режима: эмбеддинги всех ролей на устройствах пула, затем по каждому протоколу поиск по галерее и метрики."""
    splits, dropped = prepare_mode(mode, raw)
    vectors: dict[str, np.ndarray] = {}
    seconds = 0.0
    with count_progress() as progress:
        for role, split in splits.items():
            paths, _, elapsed = embed_parts(pool, split, tmp / f"{mode}_{role}", progress)
            vectors[role] = np.concatenate([np.load(path) for path in paths])
            seconds += elapsed
        results = []
        for protocol in protocols:
            gallery_rows, rows = protocol_rows(splits["val"], protocol)
            val = retrieve(splits["val"], vectors["val"], rows, vectors["val"], gallery_rows, True, progress)
            rejects = {role: retrieve(splits[role], vectors[role], np.arange(len(splits[role])), vectors["val"], gallery_rows, False, progress) for role in splits if role != "val"}
            results.append(make_result(mode, protocol, raw, splits, dropped, len(gallery_rows), val, rejects, seconds))
    return results


@click.command()
@common_options
@click.option("--weights", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=ROOT / "weights" / "tulip-so400m-14-384.ckpt", show_default=True)
@click.option("--resize-mode", type=click.Choice(RESIZE_MODES), default="squash", show_default=True, help="Как картинка приводится к 384×384: squash — родной режим TULIP, сжатие без сохранения пропорций; longest — с полями цвета фона нормализации; shortest — центральный кроп")
@click.option("--batch-size", type=click.IntRange(min=1), default=64, show_default=True)
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
    weights: Path,
    resize_mode: str,
    batch_size: int,
) -> None:
    """Замер визуальной башни TULIP на WineSensed: recall@1/3/5 и отделимость запросов без ответа. Запуск из корня: python -m scripts.bench_tulip.run.

    Галерея и запросы — val.json, два протокола, см. --protocol. loo — leave-one-out из README датасетов: каждая картинка ищет среди всех
    остальных картинок val, попаданием считается любое из прочих фото её класса. oneshot — как в бою: в галерее одно фото на класс,
    остальные фото класса идут запросами. Эмбеддинги у протоколов общие. Запросы без ответа в галерею не попадают: val_distractors.json — вина не из галереи,
    negatives.json датасета --negatives — не вино. По ним считается, насколько косинус top-1 и отрыв top-1 от top-2 отделяют их от запросов val.

    В режиме norm вход энкодера — кроп нормализации, отрендеренный по сохранённой разметке normalization.jsonl той же функцией, что в пайплайне;
    SAM3 не запускается. Картинки, где нормализация не нашла годной бутылки, до энкодера не доходят, как и в пайплайне: они выпадают из галереи
    и запросов, их число есть в таблице, а сквозной recall считает такие запросы val промахами. Поиск точный: faiss IndexFlatIP по L2-нормированным векторам.

    Несколько видеокарт в --device — параллелизм по данным: на каждой свой процесс с копией модели, картинки сплита делятся между ними поровну.
    """
    render_cfg = load_config(norm_config, [])
    winesensed, negatives = dataset_paths(datasets_dir, winesensed, negatives)
    raw = read_splits(winesensed, negatives, distractors, limit)
    with TemporaryDirectory(prefix="bench_tulip_") as tmp:
        with console.status(f"загрузка {MODEL} на {len(devices)} устр."):
            pool = DevicePool(devices, TulipWorker, (weights, resize_mode, render_cfg, batch_size, workers))
        with pool:
            console.print(f"[green]model[/] {MODEL} на {', '.join(pool.names)}, {pool.dtype}")
            results = [r for m in MODES[mode] for r in run_mode(m, PROTOCOLS[protocol], raw, pool, Path(tmp))]
            info, dtype = pool.info[0], pool.dtype
    run_info = {
        "модель": f"{MODEL}, визуальная башня, {weights.name}",
        "устройства": f"{', '.join(map(str, devices))}, {dtype}",
        "вход": f"{info['вход']}, resize_mode={resize_mode}",
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
