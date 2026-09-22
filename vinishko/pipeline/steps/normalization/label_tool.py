"""Веб-разметка целевых бутылок поверх сегментации SAM3, для калибровки отбора.

  uv run python -m vinishko.pipeline.steps.normalization.label_tool [--precompute] [--port 8765]

Сегментация идёт с настройками секции [segmentation] конфига нормализации, поэтому разметка ложится на те же
кандидаты, что видят calibrate.py и normalize.py. Считается при первом открытии картинки и кэшируется в
cache/<тег сегментации>/, --precompute считает все картинки в фоне. Разметка пишется в labels.json.

Формат labels.json: {"имя.webp": {"decision": "ok" | "no_target" | "ambiguous",
  "targets": [{"candidate": номер или null, "box": [x1, y1, x2, y2], "truncated": bool, "label_hidden": bool}],
  "updated": "..."}}.
Позитив для калибровки — бутылка, которую можно опознать по этикетке: целая или обрезанная краем кадра
(флаг truncated, ставится для статистики), с частично закрытой, но узнаваемой этикеткой. Бутылку со скрытой
или нечитаемой из-за размытия этикеткой тоже отмечают, но с флагом label_hidden: в калибровке она негатив,
а для проверки этикетки в нормализации служит эталоном. Сгенерированные картинки — no_target.
Файл review.json рядом (список имён) включает режим «только на проверку».
"""

import argparse
import contextlib
import faulthandler
import json
import mimetypes
import signal
import threading
import time
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from vinishko.pipeline.steps.normalization.seg import Segmenter, open_image

HERE = Path(__file__).resolve().parent
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DECISIONS = {"ok", "no_target", "ambiguous"}

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument("--images", type=Path, default=HERE / "images")
ap.add_argument("--labels", type=Path, default=HERE / "labels.json")
ap.add_argument("-c", "--config", type=Path, default=HERE / "normalize.toml")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument(
    "--precompute",
    action="store_true",
    help="в фоне посчитать сегментацию для всех картинок",
)
ARGS = ap.parse_args()
SC = tomllib.loads(ARGS.config.read_text())["segmentation"]
MODEL = (
    Path(SC["model"])
    if Path(SC["model"]).is_absolute()
    else ARGS.config.resolve().parent / SC["model"]
)
SEG = Segmenter(MODEL, SC["prompt"], SC["conf"], SC["imgsz"], SC["max_side"])

IMAGES = ARGS.images.resolve()
CACHE = HERE / "cache" / SEG.tag
LABELS_PATH = ARGS.labels.resolve()
CACHE.mkdir(parents=True, exist_ok=True)
LABELS: dict = json.loads(LABELS_PATH.read_text()) if LABELS_PATH.exists() else {}
_seg_lock, _label_lock = threading.Lock(), threading.Lock()
COUNTS: dict[str, int] = {}  # картинка -> число кандидатов, для режима «только трудные»
EDGE: dict[str, int] = {}  # картинка -> сколько уверенных кандидатов упираются в край кадра, для режима «обрезанные»
REVIEW_PATH = HERE / "review.json"  # имена картинок на пересмотр, режим «только на проверку»
EDGE_CONF, EDGE_MARGIN = 0.5, 0.005
EDGE_MAX_FRAC, EDGE_MAX_HEIGHT = 0.5, 0.9  # каталожные фото, где бутылка заполняет кадр или тянется на всю высоту, не считаются


def remember(name: str, res: dict) -> None:
    """Счётчики для фильтров списка по результату сегментации."""
    COUNTS[name] = len(res["cands"])
    w, h = res["w"], res["h"]
    EDGE[name] = sum(
        1
        for c in res["cands"]
        if c["conf"] >= EDGE_CONF
        and min(c["box"][0], c["box"][1], w - c["box"][2], h - c["box"][3]) < EDGE_MARGIN * max(w, h)
        and (c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1]) < EDGE_MAX_FRAC * w * h
        and c["box"][3] - c["box"][1] < EDGE_MAX_HEIGHT * h
    )


for f in CACHE.glob("*.json"):
    with contextlib.suppress(ValueError, KeyError):  # недописанный файл кэша посчитается при открытии
        remember(f.name.removesuffix(".json"), json.loads(f.read_text()))


def segment(path: Path) -> dict:
    """Кандидаты SAM3 для картинки, из кэша или свежие."""
    out = CACHE / (path.name + ".json")
    CACHE.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
        res = json.loads(out.read_text())
        remember(path.name, res)
        return res
    img = open_image(path)
    with _seg_lock:  # одна модель под локом, для разметки параллелизм не нужен
        res = {"name": path.name, **SEG(img)}
    out.write_text(json.dumps(res))
    remember(path.name, res)
    return res


def list_images() -> list[str]:
    """Имена всех картинок в папке, по алфавиту."""
    return sorted(
        p.name for p in IMAGES.iterdir() if p.suffix.lower() in EXTS and p.is_file()
    )


def image_path(name: str) -> Path | None:
    """Путь к картинке по имени; None, если имя выходит за папку или это не картинка."""
    p = IMAGES / name
    return p if p.name == name and p.suffix.lower() in EXTS and p.is_file() else None


def save_label(name: str, v: dict | None) -> dict | None:
    """Проверяет и записывает метку картинки; None удаляет метку."""
    if v is not None:
        ts = v.get("targets")
        if v.get("decision") not in DECISIONS or not isinstance(ts, list):
            raise ValueError("bad label")
        if not all(
            isinstance(t, dict)
            and isinstance(t.get("box"), list)
            and len(t["box"]) == 4
            for t in ts
        ):
            raise ValueError("bad target box")
        if (v["decision"] == "ok") != bool(ts):
            raise ValueError("targets only with decision ok")
        v = {
            "decision": v["decision"],
            "targets": [
                {
                    "candidate": t.get("candidate"),
                    "box": t["box"],
                    "truncated": bool(t.get("truncated")),
                    "label_hidden": bool(t.get("label_hidden")),
                }
                for t in ts
            ],
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    with _label_lock:
        if v is None:
            LABELS.pop(name, None)
        else:
            LABELS[name] = v
        tmp = LABELS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(LABELS, ensure_ascii=False, indent=1))
        tmp.replace(LABELS_PATH)
    return v


class Handler(BaseHTTPRequestHandler):
    """HTTP: страница, список картинок с метками, сегментация, сами картинки, запись меток."""

    def reply(
        self, code: int, body: object, ctype: str = "application/json; charset=utf-8"
    ) -> None:
        """Ответ с телом bytes как есть, иначе JSON."""
        data = (
            body
            if isinstance(body, bytes)
            else json.dumps(body, ensure_ascii=False).encode()
        )
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        """Роуты: /, /api/images, /api/seg/<имя>, /img/<имя>."""
        path = unquote(self.path.split("?")[0])
        try:
            if path == "/":
                return self.reply(
                    200,
                    (HERE / "label_tool.html").read_bytes(),
                    "text/html; charset=utf-8",
                )
            if path == "/api/images":
                return self.reply(
                    200, {
                        "images": list_images(),
                        "labels": LABELS,
                        "cands": COUNTS,
                        "edge": EDGE,
                        "review": json.loads(REVIEW_PATH.read_text()) if REVIEW_PATH.exists() else [],
                    }
                )
            for prefix in ("/api/seg/", "/img/"):
                if path.startswith(prefix):
                    p = image_path(path[len(prefix) :])
                    if p is None:
                        return self.reply(404, {"error": "not found"})
                    if prefix == "/img/":
                        return self.reply(
                            200,
                            p.read_bytes(),
                            mimetypes.guess_type(p.name)[0]
                            or "application/octet-stream",
                        )
                    return self.reply(200, segment(p))
            self.reply(404, {"error": "not found"})
        except Exception as e:
            self.reply(500, {"error": repr(e)})

    def do_POST(self) -> None:
        """Роут /api/label/<имя>: тело — метка или null."""
        path = unquote(self.path.split("?")[0])
        name = path.removeprefix("/api/label/")
        if name == path or image_path(name) is None:
            return self.reply(404, {"error": "not found"})
        try:
            body = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"null"
            )
            self.reply(200, save_label(name, body))
        except (ValueError, AttributeError) as e:
            self.reply(400, {"error": str(e)})

    def log_message(self, *a: object) -> None:
        """Без построчного лога запросов."""


def precompute() -> None:
    """Сегментация всех картинок без кэша, в фоне."""
    todo = [n for n in list_images() if n not in COUNTS]
    print(f"precompute: {len(todo)} картинок без сегментации", flush=True)
    t = time.time()
    for i, n in enumerate(todo, 1):
        try:
            segment(IMAGES / n)
        except Exception as e:
            print(f"precompute: {n}: {e!r}", flush=True)
        if i % 100 == 0 or i == len(todo):
            print(f"precompute: {i}/{len(todo)}, {time.time() - t:.0f} с", flush=True)


if __name__ == "__main__":
    faulthandler.register(
        signal.SIGUSR1, all_threads=True
    )  # kill -USR1 <pid> печатает стеки в лог, если что-то зависло
    if ARGS.precompute:
        threading.Thread(target=precompute, daemon=True).start()
    print(
        f"картинки: {IMAGES}\nразметка: {LABELS_PATH}\nсегментация: {SEG.tag}\nhttp://localhost:{ARGS.port}",
        flush=True,
    )
    ThreadingHTTPServer(("0.0.0.0", ARGS.port), Handler).serve_forever()  # noqa: S104 — доступ из Windows к WSL
