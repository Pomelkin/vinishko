import asyncio
import json
import os
from pathlib import Path

import httpx
import rich_click as click
from huggingface_hub import HfApi, get_token, hf_hub_url
from huggingface_hub.hf_api import RepoFile
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

console = Console()

HF_REPOS = {
    "winesensed": "christopher/winesensed",
    "rp2k": "JamieSJS/rp2k",
    "sop": "JamieSJS/stanford-online-products",
    "products10k": "amaye15/Products-10k",
}
OFF_API = "https://world.openfoodfacts.org/api/v2/search"
OFF_CATEGORY = "wines"
OFF_MAX_PRODUCTS = 20000
OFF_PAGE_SIZE = 100
OFF_IMAGE_KEYS = ("image_front_url", "image_packaging_url", "image_ingredients_url")
OFF_SEARCH_INTERVAL = 6.5
OFF_LIST_RETRIES, OFF_LIST_SLEEP = 5, 10.0
OFF_IMAGE_RETRIES, OFF_IMAGE_SLEEP = 5, 30.0
DATASETS = ["winesensed", "rp2k", "sop", "products10k", "off"]

USER_AGENT = "wine-scanner-hack/0.1 (hackathon; contact via github)"
CONCURRENCY = 16
CHUNK = 1 << 20


def bytes_progress() -> Progress:
    return Progress(
        TextColumn("{task.fields[name]}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def count_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[name]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    )


async def download_file(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    size: int | None,
    progress: Progress,
    dataset_task: int,
    local: bool,
    name: str,
    headers: dict | None = None,
) -> None:
    """Потоковая загрузка с докачкой; в global-режиме двигает бар датасета, в local — свой бар."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    if size and existing >= size:
        if not local:
            progress.update(dataset_task, advance=size)
        return

    headers = dict(headers or {})
    if existing:
        headers["Range"] = f"bytes={existing}-"
    task = progress.add_task("f", name=name, total=size, completed=existing) if local else dataset_task
    if not local:
        progress.update(dataset_task, advance=existing)

    async with client.stream("GET", url, headers=headers) as r:
        if r.status_code == 416:
            return
        r.raise_for_status()
        resumed = r.status_code == 206
        if existing and not resumed:
            progress.update(task, completed=0) if local else progress.update(dataset_task, advance=-existing)
        with open(dest, "ab" if resumed else "wb") as f:
            async for chunk in r.aiter_bytes(CHUNK):
                f.write(chunk)
                progress.update(task, advance=len(chunk))
    if local:
        progress.remove_task(task)


async def download_hf(client: httpx.AsyncClient, repo_id: str, out: Path, progress: Progress, local: bool) -> None:
    api = HfApi()
    files = [f for f in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True) if isinstance(f, RepoFile)]
    total = sum(f.size or 0 for f in files)
    dataset_task = progress.add_task("ds", name=repo_id, total=total, visible=not local)
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(f: RepoFile) -> None:
        async with sem:
            url = hf_hub_url(repo_id, f.path, repo_type="dataset")
            await download_file(client, url, out / f.path, f.size, progress, dataset_task, local, f.path, headers)

    await asyncio.gather(*(one(f) for f in files))
    progress.remove_task(dataset_task)




class GiveUp(Exception):
    pass


async def get_with_retry(client: httpx.AsyncClient, url: str, retries: int, sleep: float, **kwargs) -> httpx.Response:
    """GET с фиксированной паузой между попытками; после исчерпания — GiveUp."""
    for attempt in range(retries):
        try:
            r = await client.get(url, **kwargs)
            if r.status_code < 400:
                return r
            status = r.status_code
        except httpx.TransportError as e:
            status = type(e).__name__
        console.print(f"[yellow]{status}[/] {url.split('?')[0]} — attempt {attempt + 1}/{retries}, sleep {sleep:.0f}s")
        await asyncio.sleep(sleep)
    raise GiveUp(url)


async def off_products(client: httpx.AsyncClient, start_page: int):
    page = start_page
    while True:
        r = await get_with_retry(
            client,
            OFF_API,
            OFF_LIST_RETRIES,
            OFF_LIST_SLEEP,
            params={
                "categories_tags_en": OFF_CATEGORY,
                "fields": "code,product_name,brands," + ",".join(OFF_IMAGE_KEYS),
                "page_size": OFF_PAGE_SIZE,
                "page": page,
            },
        )
        products = r.json().get("products", [])
        if not products:
            return
        for p in products:
            yield p
        page += 1
        await asyncio.sleep(OFF_SEARCH_INTERVAL)


def off_jobs(products, img_dir: Path) -> list[tuple[str, Path]]:
    jobs = []
    for p in products:
        for key in OFF_IMAGE_KEYS:
            url = p.get(key)
            if url:
                stem = key.removeprefix("image_").removesuffix("_url")
                jobs.append((url, img_dir / p["code"] / f"{stem}.jpg"))
    return jobs


async def download_off(client: httpx.AsyncClient, out: Path, progress: Progress, local: bool) -> None:
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out / "products.jsonl"
    seen: dict[str, dict] = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                seen[p["code"]] = p

    start_page = len(seen) // OFF_PAGE_SIZE + 1
    listing = progress.add_task("list", name=f"OFF: listing (from page {start_page})", total=OFF_MAX_PRODUCTS, completed=len(seen))
    with open(meta_path, "a", encoding="utf-8") as meta:
        try:
            async for p in off_products(client, start_page):
                if len(seen) >= OFF_MAX_PRODUCTS:
                    break
                if p["code"] in seen:
                    continue
                seen[p["code"]] = p
                meta.write(json.dumps(p, ensure_ascii=False) + "\n")
                progress.update(listing, advance=1)
        except GiveUp:
            console.print(f"[yellow]OFF listing gave up[/] — {len(seen)} products collected, downloading their images")
    progress.remove_task(listing)

    jobs = [(u, d) for u, d in off_jobs(seen.values(), img_dir) if not d.exists()]
    dataset_task = progress.add_task("ds", name=f"OFF images ({len(jobs)} files)", total=len(jobs), visible=not local)
    sem = asyncio.Semaphore(CONCURRENCY)
    stop = asyncio.Event()

    async def one(url: str, dest: Path) -> None:
        async with sem:
            if stop.is_set():
                return
            r = await get_with_retry(client, url, OFF_IMAGE_RETRIES, OFF_IMAGE_SLEEP)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            progress.update(dataset_task, advance=1)

    tasks = [asyncio.create_task(one(u, d)) for u, d in jobs]
    try:
        await asyncio.gather(*tasks)
    except GiveUp:
        stop.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        done = sum(1 for _, d in jobs if d.exists())
        console.print(f"[yellow]OFF images gave up[/] — {done}/{len(jobs)} downloaded, rest on next run")
    progress.remove_task(dataset_task)


async def run(out: Path, local: bool, proxy: str | None) -> None:
    overall = count_progress()
    transfer = bytes_progress()
    timeout = httpx.Timeout(60, read=600)
    with Live(Group(overall, transfer), console=console, refresh_per_second=8):
        overall_task = overall.add_task("all", name="datasets", total=len(DATASETS))
        async with httpx.AsyncClient(
            proxy=proxy, follow_redirects=True, timeout=timeout, headers={"User-Agent": USER_AGENT}
        ) as client:
            for name in DATASETS:
                overall.update(overall_task, name=f"datasets · {name}")
                target = out / name
                if name in HF_REPOS:
                    await download_hf(client, HF_REPOS[name], target, transfer, local)
                else:
                    await download_off(client, target, transfer, local)
                overall.update(overall_task, advance=1)
                console.print(f"[green]done[/] {name} → {target}")


@click.command()
@click.argument("out", type=click.Path(path_type=Path))
@click.option("--pb", type=click.Choice(["global", "local"]), default="global", show_default=True, help="Бар на датасет или бар на каждый файл")
@click.option("--proxy", default=None, help="URL прокси, например http://127.0.0.1:2080")
def main(out: Path, pb: str, proxy: str | None) -> None:
    """Скачивает WineSensed, RP2K, Stanford Online Products, Products-10K и Open Food Facts (wines) в OUT."""
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
    out.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(out, pb == "local", proxy))


if __name__ == "__main__":
    main()