import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from platformdirs import user_cache_path
from rich.progress import BarColumn
from rich.progress import DownloadColumn
from rich.progress import Progress
from rich.progress import TaskID
from rich.progress import TextColumn
from rich.progress import TimeRemainingColumn
from rich.progress import TransferSpeedColumn


# Официальный https://huggingface.co/facebook/sam3 закрыт подтверждением доступа, на ModelScope тот же файл открыт
SAM3_URL = "https://modelscope.cn/models/facebook/sam3/resolve/master/sam3.pt"
SAM3_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
SAM3_FILE = "sam3.pt"

CHUNK = (
    32 << 20
)  # диапазон одного запроса; мелкие куски выравнивают нагрузку между потоками
WORKERS = 8
ATTEMPTS = 5  # оборванный диапазон докачивается с места обрыва


def cache_dir() -> Path:
    """Кэш пользователя: ~/.cache/vino на Linux, %LOCALAPPDATA%\\vino\\Cache на Windows, ~/Library/Caches/vino на macOS."""
    return user_cache_path("vino", appauthor=False)


def sam3_weights() -> Path:
    """Путь к весам SAM3 в кэше пользователя; при первом вызове скачивает их."""
    path = cache_dir() / SAM3_FILE
    if not path.is_file():
        download(SAM3_URL, path, SAM3_SHA256)
    return path


def fetch_range(
    client: httpx.Client,
    url: str,
    part: Path,
    start: int,
    end: int,
    progress: Progress,
    task: TaskID,
) -> None:
    """Пишет байты [start, end] файла по url в part на их место."""
    pos = start
    for attempt in range(ATTEMPTS):
        try:
            with (
                client.stream("GET", url, headers={"Range": f"bytes={pos}-{end}"}) as r,
                part.open("r+b") as f,
            ):
                r.raise_for_status()
                if r.status_code != httpx.codes.PARTIAL_CONTENT:
                    raise RuntimeError(
                        f"{url}: сервер не отдаёт диапазоны, код {r.status_code}"
                    )
                f.seek(pos)
                for data in r.iter_raw(1 << 20):
                    f.write(data)
                    pos += len(data)
                    progress.advance(task, len(data))
            if pos == end + 1:
                return
        except httpx.TransportError:
            if attempt == ATTEMPTS - 1:
                raise
    raise RuntimeError(
        f"{url}: диапазон {start}-{end} недокачан после {ATTEMPTS} попыток"
    )


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(16 << 20):
            h.update(data)
    return h.hexdigest()


def download(url: str, dest: Path, sha256: str) -> None:
    """Качает url в dest параллельно диапазонами по WORKERS потокам и сверяет sha256; dest появляется только целым."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    # диапазоны считаются в байтах файла, а не сжатого ответа
    headers = {"Accept-Encoding": "identity"}
    with httpx.Client(
        headers=headers, follow_redirects=True, timeout=httpx.Timeout(30, read=120)
    ) as client:
        # ModelScope редиректит на подписанную ссылку CDN: её берём один раз, чтобы потоки не ходили через редирект
        with client.stream("GET", url, headers={"Range": "bytes=0-0"}) as r:
            r.raise_for_status()
            if r.status_code != httpx.codes.PARTIAL_CONTENT:
                raise RuntimeError(
                    f"{url}: сервер не отдаёт диапазоны, код {r.status_code}"
                )
            size = int(r.headers["Content-Range"].rpartition("/")[2])
            direct = str(r.url)
        with part.open("wb") as f:
            f.truncate(size)
        columns = (
            TextColumn(f"[green]{dest.name}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        )
        with Progress(*columns) as progress, ThreadPoolExecutor(WORKERS) as pool:
            task = progress.add_task("", total=size)
            jobs = [
                pool.submit(
                    fetch_range,
                    client,
                    direct,
                    part,
                    s,
                    min(s + CHUNK, size) - 1,
                    progress,
                    task,
                )
                for s in range(0, size, CHUNK)
            ]
            try:
                for job in jobs:
                    job.result()
            except BaseException:
                for job in jobs:
                    job.cancel()  # ещё не начатые диапазоны не качаем, начатые допишутся в брошенный .part
                raise
    if (got := sha256_of(part)) != sha256:
        part.unlink()
        raise RuntimeError(f"{url}: sha256 {got}, ожидался {sha256}")
    part.replace(dest)
