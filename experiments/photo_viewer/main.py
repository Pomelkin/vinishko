import random
from functools import lru_cache
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse

PHOTO_DIR = Path(__file__).with_name("demo_photos")
SPLIT_SEED = 54
FOLDER_COUNT = 3
HOST = "127.0.0.1"
PORT = 8000

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

app = FastAPI(title="Просмотрщик фотографий")


@lru_cache(maxsize=1)
def image_groups() -> tuple[tuple[Path, ...], ...]:
    if not PHOTO_DIR.is_dir():
        return tuple(() for _ in range(FOLDER_COUNT))

    files = sorted(
        path for path in PHOTO_DIR.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    random.Random(SPLIT_SEED).shuffle(files)
    return tuple(tuple(files[index::FOLDER_COUNT]) for index in range(FOLDER_COUNT))


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.get("/api/folders")
def folders() -> list[dict[str, object]]:
    return [
        {
            "name": f"Папка {folder_index + 1}",
            "files": [
                {
                    "name": path.name,
                    "title": path.stem,
                    "url": f"/image/{folder_index}/{file_index}",
                }
                for file_index, path in enumerate(files)
            ],
        }
        for folder_index, files in enumerate(image_groups())
    ]


@app.get("/image/{folder_index}/{file_index}")
def image(folder_index: int, file_index: int) -> FileResponse:
    groups = image_groups()
    if not 0 <= folder_index < len(groups) or not 0 <= file_index < len(groups[folder_index]):
        raise HTTPException(status_code=404, detail="Изображение не найдено")
    return FileResponse(groups[folder_index][file_index])


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
    aside { display: flex; min-width: 0; flex-direction: column; border-right: 1px solid #30343c; background: #191c22; }
    .folders { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; padding: 14px; border-bottom: 1px solid #30343c; }
    button { color: inherit; border: 0; cursor: pointer; }
    .folder { padding: 10px 5px; border-radius: 9px; background: #292d35; font-weight: 700; }
    .folder.active { background: #7c5cff; }
    .folder small { display: block; margin-top: 3px; opacity: .7; font-weight: 500; }
    .list { overflow-y: auto; padding: 8px; }
    .item { display: grid; grid-template-columns: 72px minmax(0, 1fr); align-items: center; width: 100%; gap: 11px; padding: 7px; margin-bottom: 5px; border-radius: 9px; background: transparent; text-align: left; }
    .item:hover { background: #252932; }
    .item.active { background: #343946; outline: 2px solid #7c5cff; }
    .item img { width: 72px; height: 58px; border-radius: 6px; object-fit: cover; background: #0c0d10; }
    .item span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    main { display: grid; grid-template-rows: auto 1fr; min-width: 0; min-height: 0; }
    h1 { min-height: 72px; margin: 0; padding: 18px 80px; overflow: hidden; text-align: center; text-overflow: ellipsis; white-space: nowrap; font-size: clamp(24px, 3vw, 42px); }
    .stage { position: relative; display: grid; min-height: 0; place-items: center; padding: 18px 70px 30px; }
    #photo { display: block; max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 8px; box-shadow: 0 12px 40px #0008; }
    .arrow { position: absolute; top: 50%; width: 48px; height: 70px; border-radius: 12px; background: #292d35cc; font-size: 38px; transform: translateY(-50%); }
    .arrow:hover { background: #7c5cff; }
    .prev { left: 12px; }
    .next { right: 12px; }
    .empty { color: #9096a3; font-size: 20px; }
    @media (max-width: 700px) { .app { grid-template-columns: 150px 1fr; } .item { grid-template-columns: 1fr; } .item img { width: 100%; } .item span { font-size: 12px; } .folders { grid-template-columns: 1fr; } h1 { padding-inline: 20px; } }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="folders" id="folders"></div>
      <div class="list" id="list"></div>
    </aside>
    <main>
      <h1 id="title">Загрузка…</h1>
      <div class="stage" id="stage">
        <button class="arrow prev" aria-label="Предыдущее фото">‹</button>
        <img id="photo" alt="">
        <div class="empty" id="empty" hidden>В этой папке нет фотографий</div>
        <button class="arrow next" aria-label="Следующее фото">›</button>
      </div>
    </main>
  </div>
  <script>
    let folders = [], folderIndex = 0, fileIndex = 0;
    const folderBox = document.querySelector('#folders');
    const list = document.querySelector('#list');
    const title = document.querySelector('#title');
    const photo = document.querySelector('#photo');
    const empty = document.querySelector('#empty');
    const escapeText = value => value.replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char]));

    function renderFolders() {
      folderBox.innerHTML = folders.map((folder, index) =>
        `<button class="folder ${index === folderIndex ? 'active' : ''}" data-index="${index}">${folder.name}<small>${folder.files.length} фото</small></button>`
      ).join('');
    }

    function selectFolder(index) {
      folderIndex = index;
      fileIndex = 0;
      renderFolders();
      renderList();
      showPhoto();
    }

    function renderList() {
      list.innerHTML = folders[folderIndex].files.map((file, index) =>
        `<button class="item ${index === fileIndex ? 'active' : ''}" data-index="${index}"><img src="${file.url}" loading="lazy" alt=""><span title="${escapeText(file.name)}">${escapeText(file.name)}</span></button>`
      ).join('');
    }

    function showPhoto() {
      const files = folders[folderIndex].files;
      const file = files[fileIndex];
      title.textContent = file ? file.title : folders[folderIndex].name;
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
      const files = folders[folderIndex].files;
      if (!files.length) return;
      fileIndex = (fileIndex + step + files.length) % files.length;
      showPhoto();
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
    document.addEventListener('keydown', event => {
      if (event.key === 'ArrowLeft') move(-1);
      if (event.key === 'ArrowRight') move(1);
    });

    fetch('/api/folders').then(response => response.json()).then(data => {
      folders = data;
      selectFolder(0);
    });
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
