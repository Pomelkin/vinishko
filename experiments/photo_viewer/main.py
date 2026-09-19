import csv
import random
import re
from io import BytesIO
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote

import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from PIL import Image
from PIL import UnidentifiedImageError

PHOTO_DIR = Path(__file__).parents[1] / "data" / "strapi" / "img"
SAVE_DIR = Path(__file__).with_name("sorted_photos")
CATALOG_CSV = Path(__file__).parents[1] / "data" / "catalog_dataset.csv"
NEAR_DUPLICATES_CSV = Path(__file__).parents[1] / "all_candidates_reviewed.csv"
NEAR_DUPLICATE_WEIGHT = 12.0
SPLIT_SEED = 54
FOLDER_COUNT = 3
SYNC_INTERVAL_SECONDS = 20
HOST = "0.0.0.0"
PORT = 8000
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

app = FastAPI(title="Просмотрщик фотографий")


def selected_item(folder_index: int, file_index: int) -> tuple[Path, str, str]:
    groups = image_groups()
    if not 0 <= folder_index < len(groups) or not 0 <= file_index < len(groups[folder_index]):
        raise HTTPException(status_code=404, detail="Изображение не найдено")
    return groups[folder_index][file_index]


def safe_name(value: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")[:180] or "unnamed"
    if name.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                        *(f"LPT{i}" for i in range(1, 10))}:
        name = f"_{name}"
    return name


def save_unique(directory: Path, filename: str, data: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = Path(filename)
    counter = 1
    while True:
        target = directory / (filename if counter == 1 else f"{path.stem}_{counter}{path.suffix}")
        try:
            with target.open("xb") as file:
                file.write(data)
            return target
        except FileExistsError:
            counter += 1


@lru_cache(maxsize=1)
def image_groups() -> tuple[tuple[tuple[Path, str, str], ...], ...]:
    if not PHOTO_DIR.is_dir() or not CATALOG_CSV.is_file():
        return tuple(() for _ in range(FOLDER_COUNT))

    items: list[tuple[Path, str, str]] = []
    with CATALOG_CSV.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            path = PHOTO_DIR / row.get("Название фото", "")
            slug = row.get("Slug", "")
            wine_name = row.get("Название вина", "")
            if slug and path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((path, slug, wine_name))

    near_duplicates: set[str] = set()
    if NEAR_DUPLICATES_CSV.is_file():
        with NEAR_DUPLICATES_CSV.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                near_duplicates.update(filter(None, (row.get("slug_1"), row.get("slug_2"))))

    randomizer = random.Random(SPLIT_SEED)
    items.sort(
        key=lambda item: randomizer.expovariate(
            NEAR_DUPLICATE_WEIGHT if item[1] in near_duplicates else 1.0
        )
    )
    return tuple(tuple(items[index::FOLDER_COUNT]) for index in range(FOLDER_COUNT))


def labeled_slugs() -> set[str]:
    if not SAVE_DIR.is_dir():
        return set()
    folder_names = {path.name.casefold() for path in SAVE_DIR.iterdir() if path.is_dir()}
    return {
        slug
        for group in image_groups()
        for _, slug, _ in group
        if safe_name(slug).casefold() in folder_names
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML.replace("__SYNC_INTERVAL_MS__", str(SYNC_INTERVAL_SECONDS * 1000))


@app.get("/api/folders")
def folders() -> list[dict[str, object]]:
    marked = labeled_slugs()
    return [
        {
            "name": f"Папка {folder_index + 1}",
            "files": [
                {
                    "name": slug,
                    "title": slug,
                    "wine_name": wine_name,
                    "url": f"/image/{folder_index}/{file_index}",
                    "labeled": slug in marked,
                }
                for file_index, (_, slug, wine_name) in enumerate(files)
            ],
        }
        for folder_index, files in enumerate(image_groups())
    ]


@app.get("/api/labeled")
def labeled_files() -> list[str]:
    return sorted(labeled_slugs())


@app.get("/image/{folder_index}/{file_index}")
def image(folder_index: int, file_index: int) -> FileResponse:
    return FileResponse(selected_item(folder_index, file_index)[0])


@app.post("/api/save/{folder_index}/{file_index}")
async def save_image(folder_index: int, file_index: int, request: Request) -> dict[str, str]:
    _, slug, _ = selected_item(folder_index, file_index)
    original_name = unquote(request.headers.get("x-filename", ""))
    extension = Path(original_name).suffix.lower()
    if extension not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Можно загружать только изображения")

    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Файл пустой или больше 50 МБ")
    if not data:
        raise HTTPException(status_code=413, detail="Файл пустой или больше 50 МБ")
    try:
        Image.open(BytesIO(data)).verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=415, detail="Файл не является изображением") from None

    filename = f"{safe_name(Path(original_name).stem)}{extension}"
    target = save_unique(SAVE_DIR / safe_name(slug), filename, bytes(data))
    return {"message": f"Сохранено: {target.relative_to(SAVE_DIR)}"}


HTML = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Просмотрщик фотографий</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; height: 100vh; overflow: hidden; background: #111318; color: #f5f6f8; font-family: Inter, system-ui, sans-serif; }
    .app { display: grid; grid-template-columns: 320px 1fr; height: 100%; }
    aside { display: flex; min-width: 0; min-height: 0; flex-direction: column; border-right: 1px solid #30343c; background: #191c22; }
    .folders { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; padding: 14px; border-bottom: 1px solid #30343c; }
    button { color: inherit; border: 0; cursor: pointer; }
    .folder { padding: 10px 5px; border-radius: 9px; background: #292d35; font-weight: 700; }
    .folder.active { background: #7c5cff; }
    .folder small { display: block; margin-top: 3px; opacity: .7; font-weight: 500; }
    .sidebar-tools { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 9px 14px; border-bottom: 1px solid #30343c; color: #aeb4c0; font-size: 12px; }
    .sync-state { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
    .sync-state.syncing .sync-icon { animation: spin .8s linear infinite; }
    .sync-state.error { color: #e69a9a; }
    .hide-labeled { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; white-space: nowrap; }
    .hide-labeled input { accent-color: #7c5cff; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .list { overflow-y: auto; padding: 8px; }
    .item { display: grid; grid-template-columns: 72px minmax(0, 1fr); align-items: center; width: 100%; gap: 11px; padding: 7px; margin-bottom: 5px; border-radius: 9px; background: transparent; text-align: left; }
    .item:hover { background: #252932; }
    .item.active { background: #343946; outline: 2px solid #7c5cff; }
    .item.labeled { box-shadow: inset 0 0 0 2px #42c98a; }
    .item img { width: 72px; height: 64px; border-radius: 6px; object-fit: contain; background: #0c0d10; }
    .item-copy { min-width: 0; }
    .item-slug, .item-wine { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .item-wine { margin-top: 4px; color: #aeb4c0; font-size: 12px; }
    main { display: grid; grid-template-rows: auto 1fr; min-width: 0; min-height: 0; }
    header { min-width: 0; padding: 14px 80px 12px; text-align: center; }
    h1 { margin: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: clamp(24px, 3vw, 42px); }
    #wineName { margin-top: 5px; overflow: hidden; color: #aeb4c0; text-overflow: ellipsis; white-space: nowrap; font-size: clamp(14px, 1.5vw, 20px); }
    .stage { position: relative; display: grid; min-height: 0; place-items: center; padding: 18px 70px 30px; }
    .stage.dragging { outline: 5px dashed #7c5cff; outline-offset: -14px; background: #7c5cff18; }
    #photo { display: block; max-width: 82%; max-height: 78vh; object-fit: contain; border-radius: 8px; box-shadow: 0 12px 40px #0008; }
    .arrow { position: absolute; top: 50%; width: 48px; height: 70px; border-radius: 12px; background: #292d35cc; font-size: 38px; transform: translateY(-50%); }
    .arrow:hover { background: #7c5cff; }
    .prev { left: 12px; }
    .next { right: 12px; }
    .empty { color: #9096a3; font-size: 20px; }
    .notice { position: absolute; bottom: 16px; left: 50%; padding: 10px 16px; border-radius: 9px; background: #252932e8; transform: translateX(-50%); }
    @media (max-width: 700px) { .app { grid-template-columns: 150px 1fr; } .item { grid-template-columns: 1fr; } .item img { width: 100%; } .item span { font-size: 12px; } .folders { grid-template-columns: 1fr; } h1 { padding-inline: 20px; } }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="folders" id="folders"></div>
      <div class="sidebar-tools">
        <span class="sync-state" id="syncState"><span class="sync-icon">↻</span><span id="labeledStats">Размечено: 0 из 0</span></span>
        <label class="hide-labeled"><input type="checkbox" id="hideLabeled"> Скрыть размеченные</label>
      </div>
      <div class="list" id="list"></div>
    </aside>
    <main>
      <header>
        <h1 id="title">Загрузка…</h1>
        <div id="wineName"></div>
      </header>
      <div class="stage" id="stage">
        <button class="arrow prev" aria-label="Предыдущее фото">‹</button>
        <img id="photo" alt="">
        <div class="empty" id="empty" hidden>В этой папке нет фотографий</div>
        <div class="notice" id="notice" hidden></div>
        <button class="arrow next" aria-label="Следующее фото">›</button>
      </div>
    </main>
  </div>
  <script>
    let folders = [], folderIndex = 0, fileIndex = 0;
    const folderBox = document.querySelector('#folders');
    const list = document.querySelector('#list');
    const title = document.querySelector('#title');
    const wineName = document.querySelector('#wineName');
    const photo = document.querySelector('#photo');
    const empty = document.querySelector('#empty');
    const stage = document.querySelector('#stage');
    const notice = document.querySelector('#notice');
    const syncState = document.querySelector('#syncState');
    const labeledStats = document.querySelector('#labeledStats');
    const hideLabeled = document.querySelector('#hideLabeled');
    let syncing = false;
    const escapeText = value => value.replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char]));

    function renderFolders() {
      folderBox.innerHTML = folders.map((folder, index) =>
        `<button class="folder ${index === folderIndex ? 'active' : ''}" data-index="${index}">${folder.name}<small>${folder.files.length} фото</small></button>`
      ).join('');
    }

    function selectFolder(index) {
      folderIndex = index;
      fileIndex = visibleIndexes()[0] ?? 0;
      renderFolders();
      renderList();
      showPhoto();
    }

    function renderList() {
      list.innerHTML = folders[folderIndex].files.map((file, index) => ({file, index}))
        .filter(({file}) => !hideLabeled.checked || !file.labeled)
        .map(({file, index}) =>
          `<button class="item ${index === fileIndex ? 'active' : ''} ${file.labeled ? 'labeled' : ''}" data-index="${index}"><img src="${file.url}" loading="lazy" alt=""><span class="item-copy"><span class="item-slug" title="${escapeText(file.name)}">${escapeText(file.name)}</span><span class="item-wine" title="${escapeText(file.wine_name)}">${escapeText(file.wine_name)}</span></span></button>`
        ).join('');
    }

    function visibleIndexes() {
      return folders[folderIndex].files.flatMap((file, index) => hideLabeled.checked && file.labeled ? [] : [index]);
    }

    function showPhoto() {
      const files = folders[folderIndex].files;
      const selected = files[fileIndex];
      const file = selected && (!hideLabeled.checked || !selected.labeled) ? selected : null;
      title.textContent = file ? file.title : folders[folderIndex].name;
      wineName.textContent = file ? file.wine_name : '';
      photo.hidden = !file;
      empty.hidden = Boolean(file);
      if (file) {
        photo.src = file.url;
        photo.alt = file.title;
        renderList();
        list.querySelector('.item.active')?.scrollIntoView({block: 'nearest'});
      }
    }

    function move(step) {
      const indexes = visibleIndexes();
      if (!indexes.length) return;
      const position = indexes.indexOf(fileIndex);
      fileIndex = indexes[(Math.max(position, 0) + step + indexes.length) % indexes.length];
      showPhoto();
    }

    function applyLabeled(slugs) {
      const labeled = new Set(slugs);
      folders.forEach(folder => folder.files.forEach(file => file.labeled = labeled.has(file.name)));
      updateLabeledStats();
      list.querySelectorAll('.item').forEach(button => {
        const file = folders[folderIndex].files[Number(button.dataset.index)];
        button.classList.toggle('labeled', file.labeled);
        button.hidden = hideLabeled.checked && file.labeled;
      });
      const indexes = visibleIndexes();
      if (hideLabeled.checked && !indexes.includes(fileIndex)) {
        fileIndex = indexes[0] ?? 0;
        showPhoto();
      }
    }

    function updateLabeledStats() {
      const files = folders.flatMap(folder => folder.files);
      labeledStats.textContent = `Размечено: ${files.filter(file => file.labeled).length} из ${files.length}`;
    }

    async function syncLabeled() {
      if (syncing) return;
      syncing = true;
      syncState.className = 'sync-state syncing';
      syncState.title = '';
      try {
        const response = await fetch('/api/labeled');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        applyLabeled(await response.json());
        syncState.className = 'sync-state';
      } catch (error) {
        syncState.className = 'sync-state error';
        syncState.title = `Ошибка синхронизации: ${error.message}`;
      } finally {
        syncing = false;
      }
    }

    folderBox.addEventListener('click', event => {
      const button = event.target.closest('.folder');
      if (button) selectFolder(Number(button.dataset.index));
    });
    list.addEventListener('click', event => {
      const button = event.target.closest('.item');
      if (button) { fileIndex = Number(button.dataset.index); showPhoto(); }
    });
    document.querySelector('.prev').addEventListener('click', () => move(-1));
    document.querySelector('.next').addEventListener('click', () => move(1));
    hideLabeled.addEventListener('change', () => {
      const indexes = visibleIndexes();
      if (!indexes.includes(fileIndex)) fileIndex = indexes[0] ?? 0;
      renderList();
      showPhoto();
    });
    document.addEventListener('keydown', event => {
      if (event.key === 'ArrowLeft') move(-1);
      if (event.key === 'ArrowRight') move(1);
    });

    stage.addEventListener('dragover', event => {
      event.preventDefault();
      stage.classList.add('dragging');
    });
    stage.addEventListener('dragleave', () => stage.classList.remove('dragging'));
    stage.addEventListener('drop', async event => {
      event.preventDefault();
      stage.classList.remove('dragging');
      const file = event.dataTransfer.files[0];
      if (!file || !folders[folderIndex].files[fileIndex]) return;

      notice.hidden = false;
      notice.textContent = 'Сохраняю…';
      try {
        const response = await fetch(`/api/save/${folderIndex}/${fileIndex}`, {
          method: 'POST',
          headers: {'X-Filename': encodeURIComponent(file.name)},
          body: file,
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail);
        notice.textContent = result.message;
        await syncLabeled();
      } catch (error) {
        notice.textContent = `Ошибка: ${error.message}`;
      }
      setTimeout(() => notice.hidden = true, 3500);
    });

    fetch('/api/folders').then(response => response.json()).then(data => {
      folders = data;
      updateLabeledStats();
      selectFolder(0);
      setInterval(syncLabeled, __SYNC_INTERVAL_MS__);
    });
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
