import gc
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

import rich_click as click
import torch

from scripts.bench_common.cli import MODES
from scripts.bench_common.cli import Dropped
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
from scripts.bench_evie.model import ATTN_IMPLS
from scripts.bench_evie.model import HEAD_DIMS
from scripts.bench_evie.model import MODEL_ID
from scripts.bench_evie.model import embed
from scripts.bench_evie.model import load_model
from scripts.bench_evie.model import pick_attn
from scripts.bench_evie.search import retrieve_rejects
from scripts.bench_evie.search import retrieve_val
from scripts.bench_evie.store import TokenStore
from vinishko.pipeline.steps.normalization.normalize import load_config
from vinishko.pipeline.steps.vis_searcher import ColQwen3_5
from vinishko.pipeline.steps.vis_searcher import ColQwen3_5Processor

Embedded = tuple[dict[str, Split], Dropped, dict[str, TokenStore], float]
"""Режим после этапа эмбеддингов: сплиты, отсев нормализации, хранилища токенов по ролям и секунды."""


def embed_mode(
    mode: str,
    raw: dict[str, Split],
    model: ColQwen3_5,
    processor: ColQwen3_5Processor,
    render_cfg: dict,
    tmp: Path,
    dim: int,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> Embedded:
    """Эмбеддинги всех ролей одного режима во временную директорию."""
    splits, dropped = prepare_mode(mode, raw)
    if not len(query_rows(splits["val"])):
        raise click.ClickException("в val нет классов с двумя и более картинками: запросов leave-one-out не получается, увеличьте --limit")
    with count_progress() as progress:
        embedded = {role: embed(model, processor, split, render_cfg, tmp / f"{mode}_{role}.f16", dim, device, batch_size, workers, progress) for role, split in splits.items()}
    stores = {role: store for role, (store, _) in embedded.items()}
    console.print(f"[green]embedded[/] {mode}: {sum(s.nbytes for s in stores.values()) / 1e9:.1f} ГБ токенов в {tmp}")
    return splits, dropped, stores, sum(seconds for _, seconds in embedded.values())


def search_mode(mode: str, raw: dict[str, Split], embedded: Embedded, device: torch.device, dtype: torch.dtype) -> ModeResult:
    """Поиск по галерее val и метрики одного режима."""
    splits, dropped, stores, seconds = embedded
    with count_progress() as progress:
        val = retrieve_val(splits["val"], stores["val"], query_rows(splits["val"]), device, dtype, progress)
        rejects = {role: retrieve_rejects(splits[role], stores[role], stores["val"], device, dtype, progress) for role in splits if role != "val"}
    return make_result(mode, raw, splits, dropped, val, rejects, seconds)


@click.command()
@common_options
@click.option("--model", "model_id", default=MODEL_ID, show_default=True, help="Репозиторий HF либо путь до директории с весами")
@click.option("--dim", type=click.Choice([str(d) for d in HEAD_DIMS]), default="128", show_default=True, help="Ширина Prefix-MRL: сколько первых измерений токена брать. От неё линейно зависят объём токенов на диске и время поиска")
@click.option("--max-visual-tokens", type=click.IntRange(min=64), default=None, help="Потолок визуальных токенов на картинку; по умолчанию как в конфиге процессора, 1024. Фото WineSensed 480×640 дают 300")
@click.option("--attn", type=click.Choice(ATTN_IMPLS), default=None, help="Реализация внимания; по умолчанию flash_attention_2 на CUDA с bfloat16, иначе sdpa")
@click.option("--batch-size", type=click.IntRange(min=1), default=8, show_default=True)
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
    model_id: str,
    dim: str,
    max_visual_tokens: int | None,
    attn: str | None,
    batch_size: int,
) -> None:
    """Замер EVIE на WineSensed в роли поисковика картинка-к-картинке: recall@1/3/5 и отделимость запросов без ответа. Запуск из корня: python -m scripts.bench_evie.run.

    Протокол и режимы raw и norm — те же, что у scripts.bench_tulip.run: галерея и запросы val.json по схеме leave-one-out, запросы без ответа —
    val_distractors.json и negatives.json, в режиме norm вход — кроп нормализации по normalization.jsonl, отсеянные картинки выпадают.

    Отличие в модели: EVIE — поздняя интеракция, картинка кодируется не одним вектором, а вектором на каждый токен, и запрос, и галерея.
    Скор пары — MaxSim пайплайна, делённый на число токенов запроса. Поиск точный, перебором всей галереи на --device; индекс faiss тут неприменим.
    Токены пишутся во временную директорию системы и удаляются по завершении; на 128 измерениях оба режима занимают около 18 ГБ.
    Если /tmp мал либо лежит в оперативной памяти, место задаёт стандартная переменная окружения TMPDIR. Сначала считаются эмбеддинги всех режимов, затем модель выгружается
    и освобождает память устройства поиску.
    """
    if device.type == "cpu" and find_spec("fla") is not None:
        raise click.BadParameter("EVIE на cpu не запустится, пока установлен flash-linear-attention: transformers выбирает его Triton-ядро gated delta rule при импорте и зовёт на любом устройстве", param_hint="--device")
    dtype = pick_dtype(device)
    attn = attn or pick_attn(device, dtype)
    render_cfg = load_config(norm_config, [])
    raw = read_splits(winesensed, negatives, distractors, limit)
    with console.status(f"загрузка {model_id}"):
        model, processor = load_model(model_id, device, dtype, attn, int(dim), max_visual_tokens)
    console.print(f"[green]model[/] {model_id} на {device}, {dtype}, {attn}, dim {dim}")

    with TemporaryDirectory(prefix="bench_evie_") as tmp:
        embedded = {m: embed_mode(m, raw, model, processor, render_cfg, Path(tmp), int(dim), device, batch_size, workers) for m in MODES[mode]}
        del model
        gc.collect()
        torch.cuda.empty_cache()
        results = [search_mode(m, raw, embedded[m], device, dtype) for m in MODES[mode]]

    spec = ReportSpec(model=model_id.rstrip("/").split("/")[-1], score_name="MaxSim на токен", score_short="maxsim")
    run_info = {
        "модель": f"{model_id}, Prefix-MRL {dim}, двунаправленное внимание, {attn}",
        "устройство": f"{device}, {dtype}",
        "вход": f"нативное разрешение, не больше {max_visual_tokens or 1024} визуальных токенов на картинку",
        "поиск": "точный MaxSim по всей галерее, скор делится на число токенов запроса",
        "галерея и запросы": str(winesensed),
        "негативы": str(negatives),
        "конфиг нормализации": f"{norm_config}: crop={render_cfg['crop']}, background={render_cfg['background']}",
        "ограничение": "нет" if limit is None else f"--limit {limit}",
        "батч и воркеры": f"{batch_size}, {workers}",
    }
    finish(results, spec, run_info, render_cfg, examples, output)


if __name__ == "__main__":
    main()
