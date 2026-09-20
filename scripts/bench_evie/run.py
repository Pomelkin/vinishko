from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rich_click as click
import torch
from rich.progress import Progress

from scripts.bench_common.cli import MODES
from scripts.bench_common.cli import Dropped
from scripts.bench_common.cli import common_options
from scripts.bench_common.cli import console
from scripts.bench_common.cli import count_progress
from scripts.bench_common.cli import finish
from scripts.bench_common.cli import prepare_mode
from scripts.bench_common.cli import read_splits
from scripts.bench_common.data import Split
from scripts.bench_common.data import query_rows
from scripts.bench_common.metrics import ModeResult
from scripts.bench_common.metrics import Retrieval
from scripts.bench_common.metrics import make_result
from scripts.bench_common.parallel import DevicePool
from scripts.bench_common.parallel import embed_parts
from scripts.bench_common.parallel import shards
from scripts.bench_common.report import ReportSpec
from scripts.bench_evie.model import ATTN_IMPLS
from scripts.bench_evie.model import HEAD_DIMS
from scripts.bench_evie.model import MODEL_ID
from scripts.bench_evie.model import EvieWorker
from scripts.bench_evie.model import UnloadTask
from scripts.bench_evie.search import SearchTask
from scripts.bench_evie.search import gallery_blocks
from scripts.bench_evie.store import TokenStore
from scripts.bench_evie.store import join_parts
from vinishko.pipeline.steps.normalization.normalize import load_config

Embedded = tuple[dict[str, Split], Dropped, dict[str, TokenStore], float]
"""Режим после этапа эмбеддингов: сплиты, отсев нормализации, хранилища токенов по ролям и секунды."""


def embed_mode(mode: str, raw: dict[str, Split], pool: DevicePool, tmp: Path, dim: int) -> Embedded:
    """Эмбеддинги всех ролей одного режима во временную директорию: каждое устройство пишет свой кусок, куски склеиваются в порядке сплита."""
    splits, dropped = prepare_mode(mode, raw)
    if not len(query_rows(splits["val"])):
        raise click.ClickException("в val нет классов с двумя и более картинками: запросов leave-one-out не получается, увеличьте --limit")
    stores: dict[str, TokenStore] = {}
    seconds = 0.0
    with count_progress() as progress:
        for role, split in splits.items():
            stem = tmp / f"{mode}_{role}"
            parts, lengths, elapsed = embed_parts(pool, split, stem, progress)
            stores[role] = join_parts(parts, lengths, stem.with_suffix(".f16"), dim)
            seconds += elapsed
    console.print(f"[green]embedded[/] {mode}: {sum(s.nbytes for s in stores.values()) / 1e9:.1f} ГБ токенов в {tmp}")
    return splits, dropped, stores, seconds


def retrieve(pool: DevicePool, split: Split, gallery: TokenStore, queries: TokenStore, rows: np.ndarray, exclude_self: bool, progress: Progress) -> Retrieval:
    """Точный MaxSim запросов rows со всей галереей; запросы делятся между устройствами, галерею каждое проходит целиком."""
    tasks = [SearchTask(gallery, queries, rows[start:stop], exclude_self) if stop > start else None for start, stop in shards(len(rows), len(pool))]
    bar = progress.add_task(split.role, name=f"поиск {split.role}", total=len(rows) * len(gallery_blocks(gallery, pool.dtype)))
    found = [r for r in pool.map(tasks, lambda n: progress.update(bar, advance=n)) if r is not None]
    return Retrieval(split, rows, np.concatenate([scores for scores, _ in found]), np.concatenate([ids for _, ids in found]))


def search_mode(mode: str, raw: dict[str, Split], embedded: Embedded, pool: DevicePool) -> ModeResult:
    """Поиск по галерее val и метрики одного режима."""
    splits, dropped, stores, seconds = embedded
    with count_progress() as progress:
        val = retrieve(pool, splits["val"], stores["val"], stores["val"], query_rows(splits["val"]), True, progress)
        rejects = {role: retrieve(pool, splits[role], stores["val"], stores[role], np.arange(len(splits[role])), False, progress) for role in splits if role != "val"}
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
    devices: list[torch.device],
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

    Несколько видеокарт в --device — параллелизм по данным: на каждой свой процесс с копией модели, картинки сплита, а на поиске запросы, делятся между ними поровну.
    """
    if devices[0].type == "cpu" and find_spec("fla") is not None:
        raise click.BadParameter("EVIE на cpu не запустится, пока установлен flash-linear-attention: transformers выбирает его Triton-ядро gated delta rule при импорте и зовёт на любом устройстве", param_hint="--device")
    render_cfg = load_config(norm_config, [])
    raw = read_splits(winesensed, negatives, distractors, limit)
    with TemporaryDirectory(prefix="bench_evie_") as tmp:
        with console.status(f"загрузка {model_id} на {len(devices)} устр."):
            pool = DevicePool(devices, EvieWorker, (model_id, attn, int(dim), max_visual_tokens, render_cfg, batch_size, workers))
        with pool:
            dtype, attn = pool.dtype, pool.info[0]["attn"]
            console.print(f"[green]model[/] {model_id} на {', '.join(pool.names)}, {dtype}, {attn}, dim {dim}")
            embedded = {m: embed_mode(m, raw, pool, Path(tmp), int(dim)) for m in MODES[mode]}
            pool.map([UnloadTask()] * len(pool), lambda _: None)
            results = [search_mode(m, raw, embedded[m], pool) for m in MODES[mode]]

    spec = ReportSpec(model=model_id.rstrip("/").split("/")[-1], score_name="MaxSim на токен", score_short="maxsim")
    run_info = {
        "модель": f"{model_id}, Prefix-MRL {dim}, двунаправленное внимание, {attn}",
        "устройства": f"{', '.join(map(str, devices))}, {dtype}",
        "вход": f"нативное разрешение, не больше {max_visual_tokens or 1024} визуальных токенов на картинку",
        "поиск": "точный MaxSim по всей галерее, скор делится на число токенов запроса",
        "галерея и запросы": str(winesensed),
        "негативы": str(negatives),
        "конфиг нормализации": f"{norm_config}: crop={render_cfg['crop']}, background={render_cfg['background']}",
        "ограничение": "нет" if limit is None else f"--limit {limit}",
        "батч и воркеры на устройство": f"{batch_size}, {workers}",
    }
    finish(results, spec, run_info, render_cfg, examples, output)


if __name__ == "__main__":
    main()
