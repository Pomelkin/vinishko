from pathlib import Path

import rich_click as click
import torch
from torch import nn

from scripts.bench_common.cli import MODES
from scripts.bench_common.cli import ROOT
from scripts.bench_common.cli import common_options
from scripts.bench_common.cli import console
from scripts.bench_common.cli import count_progress
from scripts.bench_common.cli import finish
from scripts.bench_common.cli import pick_dtype
from scripts.bench_common.cli import prepare_mode
from scripts.bench_common.cli import read_splits
from scripts.bench_common.data import Split
from scripts.bench_common.data import query_rows
from scripts.bench_common.metrics import ModeResult
from scripts.bench_common.metrics import make_result
from scripts.bench_common.report import ReportSpec
from scripts.bench_tulip.model import MODEL
from scripts.bench_tulip.model import RESIZE_MODES
from scripts.bench_tulip.model import Preprocess
from scripts.bench_tulip.model import embed
from scripts.bench_tulip.model import load_encoder
from scripts.bench_tulip.search import retrieve_rejects
from scripts.bench_tulip.search import retrieve_val
from vinishko.pipeline.steps.normalization.normalize import load_config

SPEC = ReportSpec(model=MODEL, score_name="косинус", score_short="cos")


def run_mode(
    mode: str,
    raw: dict[str, Split],
    visual: nn.Module,
    preprocess: Preprocess,
    render_cfg: dict,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    workers: int,
) -> ModeResult:
    """Замер одного режима: эмбеддинги всех ролей, поиск по галерее val, метрики."""
    splits, dropped = prepare_mode(mode, raw)
    rows = query_rows(splits["val"])
    if not len(rows):
        raise click.ClickException("в val нет классов с двумя и более картинками: запросов leave-one-out не получается, увеличьте --limit")
    with count_progress() as progress:
        embedded = {role: embed(visual, preprocess, split, render_cfg, device, dtype, batch_size, workers, progress) for role, split in splits.items()}
        gallery = embedded["val"][0]
        val = retrieve_val(splits["val"], gallery, rows, progress)
        rejects = {role: retrieve_rejects(splits[role], embedded[role][0], gallery, progress) for role in splits if role != "val"}
    return make_result(mode, raw, splits, dropped, val, rejects, sum(seconds for _, seconds in embedded.values()))


@click.command()
@common_options
@click.option("--weights", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=ROOT / "weights" / "tulip-so400m-14-384.ckpt", show_default=True)
@click.option("--resize-mode", type=click.Choice(RESIZE_MODES), default="squash", show_default=True, help="Как картинка приводится к 384×384: squash — родной режим TULIP, сжатие без сохранения пропорций; longest — с полями цвета фона нормализации; shortest — центральный кроп")
@click.option("--batch-size", type=click.IntRange(min=1), default=64, show_default=True)
def main(
    output: Path,
    device: torch.device,
    mode: str,
    winesensed: Path,
    negatives: Path,
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

    Протокол — из README датасетов. Галерея и запросы — val.json, leave-one-out: каждая картинка ищет среди всех остальных картинок val,
    запросами идут картинки классов, где их две и больше. Запросы без ответа в галерею не попадают: val_distractors.json — вина не из галереи,
    negatives.json датасета --negatives — не вино. По ним считается, насколько косинус top-1 и отрыв top-1 от top-2 отделяют их от запросов val.

    В режиме norm вход энкодера — кроп нормализации, отрендеренный по сохранённой разметке normalization.jsonl той же функцией, что в пайплайне;
    SAM3 не запускается. Картинки, где нормализация не нашла годной бутылки, до энкодера не доходят, как и в пайплайне: они выпадают из галереи
    и запросов, их число есть в таблице, а сквозной recall считает такие запросы val промахами. Поиск точный: faiss IndexFlatIP по L2-нормированным векторам.
    """
    dtype = pick_dtype(device)
    render_cfg = load_config(norm_config, [])
    raw = read_splits(winesensed, negatives, distractors, limit)
    with console.status(f"загрузка {MODEL}"):
        visual, preprocess, size = load_encoder(weights, device, dtype, resize_mode, tuple(render_cfg["background"]["color"]))
    console.print(f"[green]model[/] {MODEL} на {device}, {dtype}")

    results = [run_mode(m, raw, visual, preprocess, render_cfg, device, dtype, batch_size, workers) for m in MODES[mode]]
    run_info = {
        "модель": f"{MODEL}, визуальная башня, {weights.name}",
        "устройство": f"{device}, {dtype}",
        "вход": f"{size[0]}×{size[1]}, resize_mode={resize_mode}",
        "поиск": "faiss IndexFlatIP, точный, по L2-нормированным векторам",
        "галерея и запросы": str(winesensed),
        "негативы": str(negatives),
        "конфиг нормализации": f"{norm_config}: crop={render_cfg['crop']}, background={render_cfg['background']}",
        "ограничение": "нет" if limit is None else f"--limit {limit}",
        "батч и воркеры": f"{batch_size}, {workers}",
    }
    finish(results, SPEC, run_info, render_cfg, examples, output)


if __name__ == "__main__":
    main()
