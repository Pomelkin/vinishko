import queue
import signal
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from multiprocessing import get_context
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any
from typing import Protocol

import rich_click as click
import torch
from rich.progress import Progress

from scripts.bench_common.data import Split

Advance = Callable[[int], None]
"""Воркер сообщает, на сколько штук продвинулся; бар рисует главный процесс."""


class Handler(Protocol):
    """Состояние воркера на одном устройстве: модель грузится один раз в конструкторе, дальше приходят задачи."""

    info: dict[str, str]
    """Что воркер выбрал сам по своему устройству; ключ dtype обязателен."""

    def __call__(self, task: Any, advance: Advance) -> Any:
        """Выполняет задачу и возвращает результат; оба пересылаются между процессами, поэтому должны сериализоваться."""
        ...


@dataclass(frozen=True)
class EmbedTask:
    """Посчитать эмбеддинги части сплита и записать их в path."""

    split: Split
    path: Path


def pick_dtype(device: torch.device) -> torch.dtype:
    """bfloat16, если устройство считает в нём аппаратно, иначе float32. Зовётся в воркере: проверка поднимает контекст CUDA.

    У CUDA это Ampere и новее, программная эмуляция не в счёт. У CPU — инструкции AVX512-BF16 либо AMX:
    без них bfloat16 на процессоре работает, но медленнее float32.
    """
    if device.type == "cuda":
        supported = torch.cuda.is_bf16_supported(including_emulation=False)
    else:
        supported = (
            torch.cpu._is_avx512_bf16_supported() or torch.cpu._is_amx_tile_supported()
        )
    return torch.bfloat16 if supported else torch.float32


def worker(
    device: str,
    factory: Callable[..., Handler],
    args: tuple,
    tasks: Queue,
    results: Queue,
) -> None:
    """Процесс одного устройства. Сообщения в results: ready с info, progress, done с результатом задачи, fatal с текстом ошибки. None в tasks завершает."""
    signal.signal(
        signal.SIGINT, signal.SIG_IGN
    )  # Ctrl+C обрабатывает главный процесс и сам гасит воркеры
    try:
        target = torch.device(device)
        if target.type == "cuda":
            torch.cuda.set_device(
                target
            )  # ядра Triton и flash-attn запускаются на текущем устройстве, а не на устройстве тензора
        handler = factory(target, *args)
    except Exception:
        results.put(("fatal", device, traceback.format_exc()))
        return
    results.put(("ready", device, handler.info))
    while (task := tasks.get()) is not None:
        try:
            result = handler(task, lambda n: results.put(("progress", device, n)))
        except Exception:
            results.put(("fatal", device, traceback.format_exc()))
            return
        results.put(("done", device, result))


class DevicePool:
    """По процессу на устройство, у каждого своя копия модели. Главный процесс CUDA не трогает и памяти на картах не занимает.

    Процессы не daemon: им нужны собственные дочерние воркеры даталоадера. Поэтому пул обязательно закрывать, удобнее через with.
    """

    def __init__(
        self, devices: list[torch.device], factory: Callable[..., Handler], args: tuple
    ) -> None:
        ctx = get_context("spawn")  # fork с CUDA несовместим
        self.names = [str(d) for d in devices]
        self.results: Queue = ctx.Queue()
        self.tasks: dict[str, Queue] = {name: ctx.Queue() for name in self.names}
        self.procs = [
            ctx.Process(
                target=worker,
                args=(name, factory, args, self.tasks[name], self.results),
                name=name,
            )
            for name in self.names
        ]
        for proc in self.procs:
            proc.start()
        try:
            ready = dict(self.message()[1:] for _ in self.procs)
        except BaseException:
            self.close(force=True)
            raise
        self.info: list[dict[str, str]] = [ready[name] for name in self.names]

    def __len__(self) -> int:
        return len(self.names)

    def __enter__(self) -> "DevicePool":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *_: object) -> None:
        self.close(force=exc_type is not None)

    @property
    def dtype(self) -> torch.dtype:
        """Общий тип вычислений; устройства разных поколений дали бы несравнимые половины одного замера."""
        found = {info["dtype"] for info in self.info}
        if len(found) > 1:
            raise click.ClickException(
                f"устройства выбрали разные типы вычислений: {dict(zip(self.names, self.info, strict=True))}"
            )
        return getattr(torch, found.pop().removeprefix("torch."))

    def message(self) -> tuple[str, str, Any]:
        """Следующее сообщение воркеров. Прогон падает, если воркер прислал fatal либо умер молча, например его убил OOM-killer."""
        while True:
            try:
                kind, device, payload = self.results.get(timeout=5)
            except queue.Empty:
                dead = [
                    f"{p.name}: код {p.exitcode}"
                    for p in self.procs
                    if not p.is_alive()
                ]
                if dead:
                    raise RuntimeError(f"воркеры завершились: {dead}") from None
                continue
            if kind == "fatal":
                raise RuntimeError(f"воркер {device} упал:\n{payload}")
            return kind, device, payload

    def map(self, tasks: list[Any], advance: Advance) -> list[Any]:
        """Задача i уходит устройству i, None — устройству делать нечего. Возвращает результаты в том же порядке, у пропущенных None."""
        pending = set()
        for name, task in zip(self.names, tasks, strict=True):
            if task is not None:
                self.tasks[name].put(task)
                pending.add(name)
        done: dict[str, Any] = {}
        while pending:
            kind, device, payload = self.message()
            if kind == "progress":
                advance(payload)
            else:
                done[device] = payload
                pending.discard(device)
        return [done.get(name) for name in self.names]

    def close(self, force: bool = False) -> None:
        """Гасит воркеры: штатно просит их выйти, при ошибке в главном процессе убивает сразу."""
        if not force:
            for tasks in self.tasks.values():
                tasks.put(None)
            for proc in self.procs:
                proc.join(timeout=30)
        for proc in self.procs:
            if proc.is_alive():
                proc.terminate()
        for proc in self.procs:
            proc.join()


def shards(total: int, parts: int) -> list[tuple[int, int]]:
    """Границы parts подряд идущих кусков примерно равной длины."""
    edges = [total * k // parts for k in range(parts + 1)]
    return list(pairwise(edges))


def embed_parts(
    pool: DevicePool, split: Split, stem: Path, progress: Progress
) -> tuple[list[Path], list[Any], float]:
    """Делит сплит между устройствами подряд идущими кусками. Возвращает файлы кусков по порядку, ответы воркеров и время этапа в секундах."""
    tasks = [
        EmbedTask(split.part(start, stop), stem.with_name(f"{stem.name}.part{k}"))
        if stop > start
        else None
        for k, (start, stop) in enumerate(shards(len(split), len(pool)))
    ]
    bar = progress.add_task(
        split.role, name=f"эмбеддинги {split.role}", total=len(split)
    )
    started = time.perf_counter()
    results = pool.map(tasks, lambda n: progress.update(bar, advance=n))
    return (
        [t.path for t in tasks if t is not None],
        [r for r in results if r is not None],
        time.perf_counter() - started,
    )
