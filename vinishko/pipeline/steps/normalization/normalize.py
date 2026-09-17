import json
import math
import re
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rich_click as click
from PIL import Image

from vinishko.pipeline.steps.normalization.features import candidate_features, gray_small
from vinishko.pipeline.steps.normalization.seg import Segmenter, label_stats, open_image

HERE = Path(__file__).resolve().parent
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BORDERS = {
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT_101,
    "constant": cv2.BORDER_CONSTANT,
}


# ---------- конфиг ----------


def load_config(path: Path, overrides: list[str]) -> dict:
    """Конфиг из TOML с переопределениями вида секция.ключ=значение."""
    cfg = tomllib.loads(path.read_text())
    for item in overrides:
        key, _, raw = item.partition("=")
        section, _, name = key.strip().partition(".")
        if section not in cfg or name not in cfg[section]:
            raise SystemExit(f"--set {item}: нет параметра {key} в {path.name}")
        try:
            value = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw  # строку можно писать без кавычек
        cfg[section][name] = value
    return cfg


def resolve(p: str, base: Path) -> Path:
    """Относительный путь считается от base."""
    return Path(p) if Path(p).is_absolute() else base / p


# ---------- отбор бутылок ----------


def score_candidates(gray: np.ndarray, seg: dict, calib: dict) -> list[float]:
    """Вероятность «целевая бутылка» для каждого кандидата по логрегу из калибровки."""
    feats = candidate_features(gray, seg)
    if not feats:
        return []
    X = np.array([[f[k] for k in calib["features"]] for f in feats])
    z = (X - np.array(calib["mean"])) / np.array(calib["std"])
    logit = z @ np.array(calib["coef"]) + calib["intercept"]
    return (1 / (1 + np.exp(-logit))).tolist()


# ---------- геометрия ----------


def all_points(polys: list) -> np.ndarray:
    """Все вершины полигонов одним массивом (N, 2)."""
    return np.concatenate([np.asarray(p, np.float64) for p in polys if len(p) >= 3])


def neck_and_base(pts: np.ndarray, center: np.ndarray, axis: np.ndarray, orient: dict) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Концы оси: какой из них горлышко. Возвращает (горлышко, дно, способ, контраст профиля ширины)."""
    t = (pts - center) @ axis
    hist, edges = np.histogram(t, bins=20)
    widths = hist / max(edges[1] - edges[0], 1e-9)
    a_end, b_end = widths[:3].mean(), widths[-3:].mean()  # ширина у концов «минус» и «плюс» оси
    contrast = float(abs(a_end - b_end) / (np.percentile(widths, 90) + 1e-9))
    p_minus, p_plus = center + axis * t.min(), center + axis * t.max()
    tilt = math.degrees(math.acos(min(1.0, abs(axis[1]))))  # отклонение оси от вертикали, 0..90
    upper = (p_minus, p_plus) if p_minus[1] < p_plus[1] else (p_plus, p_minus)
    if tilt <= orient["upright_within_deg"]:
        # профиль ширины путает дно с горлышком при дырах и бликах в маске; почти вертикальную бутылку считаем стоящей
        return *upper, "upright_prior", contrast
    if orient["neck"] == "profile" and contrast >= orient["min_profile_contrast"]:
        ends = (p_minus, p_plus) if a_end < b_end else (p_plus, p_minus)
        return *ends, "profile", contrast
    return *upper, "up", contrast


def bottle_axis(polys: list, orient: dict) -> dict:
    """Ось бутылки по маске: центр, точки горлышка и дна, угол поворота до вертикали горлышком вверх.

    Плюс система координат бутылки: единичные векторы вдоль (u, от горлышка к дну) и поперёк (n) оси,
    видимая длина и ширина корпуса — 90-й перцентиль размаха маски поперёк оси по 30 срезам.
    """
    allp = all_points(polys)
    (x1, y1), (x2, y2) = allp.min(0), allp.max(0)
    s = 400 / max(x2 - x1, y2 - y1, 1)  # считаем на маске ~400 px, этого хватает
    m = np.zeros((int((y2 - y1) * s) + 3, int((x2 - x1) * s) + 3), np.uint8)
    cv2.fillPoly(m, [np.round((np.asarray(p, np.float64) - [x1, y1]) * s).astype(np.int32) + 1 for p in polys if len(p) >= 3], 1)
    ys, xs = np.nonzero(m)
    pts = (np.stack([xs, ys], 1).astype(np.float64) - 1) / s + [x1, y1]
    center = pts.mean(0)
    axis = np.linalg.eigh(np.cov((pts - center).T))[1][:, -1]
    neck, base, method, contrast = neck_and_base(pts, center, axis, orient)
    length = float(np.linalg.norm(base - neck))
    u = (base - neck) / max(length, 1e-9)
    n = np.array([-u[1], u[0]])
    # cv2.getRotationMatrix2D(angle=a+90) переводит направление с углом a (ось y вниз) в (0, -1), то есть вверх
    angle = math.degrees(math.atan2(-u[1], -u[0])) + 90 if orient["enabled"] else 0.0
    w = (pts - neck) @ n
    bins = np.clip(((pts - neck) @ u / max(length, 1e-9) * 30).astype(int), 0, 29)
    spans = [np.ptp(w[bins == i]) if (bins == i).sum() > 3 else 0.0 for i in range(30)]
    return {
        "center": center,
        "neck": neck,
        "base": base,
        "angle": angle,
        "method": method,
        "contrast": contrast,
        "u": u,
        "n": n,
        "length": length,
        "body_width": float(np.percentile(spans, 90)),
    }


def label_geometry(ax: dict, polys: list) -> dict:
    """Положение этикетки вдоль бутылки (доля полной длины от горлышка) и её размеры в системе бутылки."""
    lp = all_points(polys) - ax["neck"]
    full = max(ax["length"], 3.3 * ax["body_width"])  # корпус может быть скрыт: полная длина не меньше 3.3 ширины
    t, w = lp @ ax["u"], lp @ ax["n"]
    return {
        "position": float(t.mean() / max(full, 1e-9)),
        "along_px": float(np.ptp(t)),
        "across_px": float(np.ptp(w)),
        "width_frac": float(np.ptp(w) / max(ax["body_width"], 1e-9)),
    }


def affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Применить аффинную матрицу 2x3 к точкам (N, 2)."""
    return pts @ M[:, :2].T + M[:, 2]


def fast_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Гауссово размытие через уменьшенную копию."""
    # размываем уменьшенную копию, на больших кропах честный Гаусс с sigma ~100 px идёт секундами
    h, w = img.shape[:2]
    f = max(1.0, sigma / 3)
    small = cv2.resize(
        img, (max(1, round(w / f)), max(1, round(h / f))), interpolation=cv2.INTER_AREA
    )
    small = cv2.GaussianBlur(small, (0, 0), sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def rotated_extent(polys: list, center: np.ndarray, angle: float) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Матрица поворота вокруг центра маски и габариты маски после поворота: (M, (x1, y1, ширина, высота))."""
    M = cv2.getRotationMatrix2D(tuple(map(float, center)), angle, 1.0)
    verts = affine(M, all_points(polys))
    (rx1, ry1), (rx2, ry2) = verts.min(0), verts.max(0)
    return M, (rx1, ry1, rx2 - rx1, ry2 - ry1)


Neighbor = tuple[int, tuple[float, float, float, float]]  # номер кандидата и его бокс в системе кропа


def limit_padding(own: tuple[float, float, float, float], others: list[Neighbor], px: float, py: float) -> tuple[float, float, list[int]]:
    """Урезает запас, чтобы он не заходил в боксы соседей. Боксы (x1, y1, x2, y2) в системе координат кропа.

    Сосед, перекрывающий бокс самой бутылки, не учитывается: запасом это не исправить.
    Запас по оси урезается симметрично, чтобы бутылка оставалась по центру кропа.
    Возвращает px, py и номера соседей, которые урезали запас.
    """
    x1, y1, x2, y2 = own
    gaps_x = [(x1 - ox2 if ox2 <= x1 else ox1 - x2, oid) for oid, (ox1, oy1, ox2, oy2) in others
              if oy2 > y1 - py and oy1 < y2 + py and (ox2 <= x1 or ox1 >= x2)]  # соседи сбоку на уровне кропа
    px, by_x = min([(px, -1), *((max(0.0, g), oid) for g, oid in gaps_x)])
    gaps_y = [(y1 - oy2 if oy2 <= y1 else oy1 - y2, oid) for oid, (ox1, oy1, ox2, oy2) in others
              if ox2 > x1 - px and ox1 < x2 + px and (oy2 <= y1 or oy1 >= y2)]  # сверху и снизу, с уже урезанным px
    py, by_y = min([(py, -1), *((max(0.0, g), oid) for g, oid in gaps_y)])
    return px, py, sorted({i for i in (by_x, by_y) if i >= 0})


def render_bottle(
    rgb: np.ndarray,
    polys: list,
    cfg: dict,
    angle: float | None = None,
    shift: tuple[float, float] = (0.0, 0.0),
    others: list[tuple[int, list]] | None = None,
) -> tuple[np.ndarray, dict]:
    """Кроп одной бутылки: поворот, запас, блюр фона. Возвращает картинку и описание преобразования.

    angle переопределяет угол, найденный по оси маски; shift сдвигает окно кропа в долях ширины и высоты бутылки.
    Оба нужны даталоадеру для джиттера поверх сохранённой разметки; запас и фон джиттерятся через cfg.
    others — [(номер кандидата, полигоны)] соседних бутылок: при crop.avoid_neighbors запас не заходит в их боксы.
    """
    ax = bottle_axis(polys, cfg["orientation"])
    if angle is None:
        angle = ax["angle"]
    M, (rx1, ry1, bw, bh) = rotated_extent(polys, ax["center"], angle)
    px, py = cfg["crop"]["padding_x"] * bw, cfg["crop"]["padding_y"] * bh
    limited_by: list[int] = []
    if cfg["crop"]["avoid_neighbors"] and others:
        boxes = []
        for oid, opolys in others:  # соседей поворачиваем той же матрицей, что и бутылку
            ov = affine(M, all_points(opolys))
            boxes.append((oid, (*ov.min(0), *ov.max(0))))
        px, py, limited_by = limit_padding((rx1, ry1, rx1 + bw, ry1 + bh), boxes, px, py)
    W, H = max(1, math.ceil(bw + 2 * px)), max(1, math.ceil(bh + 2 * py))
    A = M.copy()
    A[:, 2] -= [rx1 - px + shift[0] * bw, ry1 - py + shift[1] * bh]
    border = BORDERS[cfg["crop"]["border"]]
    crop = cv2.warpAffine(rgb, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=border, borderValue=(0, 0, 0))
    inside = cv2.warpAffine(np.ones(rgb.shape[:2], np.uint8), A, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)

    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [np.round(affine(A, np.asarray(p, np.float64))).astype(np.int32) for p in polys if len(p) >= 3], 255)
    side = max(W, H)
    bg = cfg["background"]
    if bg["mode"] != "none":
        r = round(bg["dilate"] * side)
        if r > 0:
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        alpha = mask.astype(np.float32) / 255
        if bg["feather"] > 0:
            alpha = cv2.GaussianBlur(alpha, (0, 0), bg["feather"] * side)
        back = fast_blur(crop, bg["blur_sigma"] * side) if bg["mode"] == "blur" else np.full_like(crop, bg["color"])
        crop = (alpha[..., None] * crop + (1 - alpha[..., None]) * back).round().astype(np.uint8)

    out_h = cfg["output"]["height"]
    if out_h and out_h != H:
        k = out_h / H
        crop = cv2.resize(crop, (max(1, round(W * k)), out_h), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
        A = A * k
    info = {
        "angle_deg": round(angle, 2),
        "neck_method": ax["method"],
        "neck_profile_contrast": round(ax["contrast"], 3),
        "neck_point": [round(v, 1) for v in ax["neck"]],
        "base_point": [round(v, 1) for v in ax["base"]],
        "bottle_size_px": [round(bw, 1), round(bh, 1)],
        "padding_px": [round(px, 1), round(py, 1)],
        "padding_limited_by": limited_by,
        "out_of_bounds_frac": round(1 - float(inside.mean()), 4),
        "matrix_src_to_dst": np.round(A, 6).tolist(),
        "matrix_dst_to_src": np.round(cv2.invertAffineTransform(A), 6).tolist(),
        "width": crop.shape[1],
        "height": crop.shape[0],
    }
    return crop, info


# ---------- этикетка: маска и вырезка ----------


def polys_box(polys: list) -> list[int]:
    """Бокс [x1, y1, x2, y2] включительно по вершинам полигонов."""
    pts = all_points(polys)
    (x1, y1), (x2, y2) = pts.min(0), pts.max(0)
    return [int(x1), int(y1), int(x2), int(y2)]


def polys_mask(polys: list, box: list[int]) -> np.ndarray:
    """bool-маска размером с бокс включительно: mask[y - y1, x - x1] для пикселя оригинала (x, y)."""
    x1, y1, x2, y2 = box
    m = np.zeros((y2 - y1 + 1, x2 - x1 + 1), np.uint8)
    cv2.fillPoly(m, [np.asarray(p, np.int32) - [x1, y1] for p in polys if len(p) >= 3], 1)
    return m.astype(bool)


def render_label(rgb: np.ndarray, polys: list, angle: float, padding: float) -> np.ndarray:
    """Вырезка этикетки под OCR: оригинал в полном разрешении, поворот как у бутылки (текст горизонтально).

    Прямоугольник вокруг маски с запасом. Без блюра и маскирования, чтобы не срезать края букв.
    """
    pts = all_points(polys)
    M = cv2.getRotationMatrix2D(tuple(map(float, pts.mean(0))), angle, 1.0)
    v = affine(M, pts)
    (x1, y1), (x2, y2) = v.min(0), v.max(0)
    px, py = padding * (x2 - x1), padding * (y2 - y1)
    W, H = max(1, math.ceil(x2 - x1 + 2 * px)), max(1, math.ceil(y2 - y1 + 2 * py))
    A = M.copy()
    A[:, 2] -= [x1 - px, y1 - py]
    return cv2.warpAffine(rgb, A, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def save_image(path: Path, rgb: np.ndarray, fmt: str, quality: int) -> None:
    """RGB-массив в файл; неудачная запись — ошибка, а не молчаливый пропуск."""
    params = {"jpg": [cv2.IMWRITE_JPEG_QUALITY, quality], "webp": [cv2.IMWRITE_WEBP_QUALITY, quality], "png": []}[fmt]
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), params):
        raise OSError(f"не удалось записать {path}")


# ---------- прогон ----------

IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


@dataclass
class Crop:
    """Кроп одной бутылки: RGB-массив и как он получен из оригинала (матрицы в info).

    mode: "bottle" — поворот, запас и заглушённый фон; "bbox" — маска ненадёжна, bbox без поворота и блюра;
    "full_image" — бутылка не найдена, целое фото. label_polys — маска главной этикетки в пикселях оригинала,
    если label_status == "ok".
    """

    image: np.ndarray
    index: int
    score: float | None
    box: list[int]
    info: dict
    mode: str = "bottle"
    label_status: str = "unchecked"
    label_polys: list | None = None


class Normalizer:
    """Сегментация → отбор по калибровке → ось бутылки и этикетка → кропы. Разметка без рендера — annotate, кропы — вызов."""

    def __init__(self, cfg: dict, base: Path) -> None:
        """Читает калибровку и поднимает сегментатор; base — папка, от которой считаются пути в cfg."""
        self.cfg, self.base = cfg, base
        sel, sc, lb = cfg["selection"], cfg["segmentation"], cfg["label"]
        self.calib = json.loads(resolve(sel["calibration"], base).read_text())
        self.threshold = sel["threshold"] if sel["threshold"] >= 0 else self.calib["threshold"]
        self.seg = Segmenter(
            resolve(sc["model"], base),
            sc["prompt"],
            sc["conf"],
            sc["imgsz"],
            sc["max_side"],
            label_prompt=lb["prompt"] or None,
            label_conf=lb["min_conf"],
            exclude_prompts=lb["exclude_prompts"],
        )
        if self.seg.tag != self.calib.get("segmentation"):
            print(
                f"внимание: калибровка сделана для сегментации {self.calib.get('segmentation')}, сейчас {self.seg.tag}",
                file=sys.stderr,
            )

    def largest_label(self, labels: list[dict], ax: dict) -> list | None:
        """Полигоны самой крупной этикетки корпуса; полоски уже min_width_frac ширины бутылки (акцизные марки) не считаются."""
        min_frac = self.cfg["label_crop"]["min_width_frac"]
        polys = [lab["polys"] for lab in labels if lab["polys"] and label_geometry(ax, lab["polys"])["width_frac"] >= min_frac]
        return max(polys, key=lambda ps: sum(cv2.contourArea(np.asarray(p, np.float32)) for p in ps), default=None)

    def check_label(self, cand: dict, ax: dict) -> tuple[str, list | None]:
        """("ok", полигоны главной этикетки корпуса) или (причина, None).

        Порядок: есть ли этикетка, видна ли, резкая ли, виден ли корпус, где этикетка и какого она размера.
        """
        lb, stats = self.cfg["label"], cand["label"]
        main = self.largest_label(cand["label_polys"], ax)
        geo = label_geometry(ax, main) if main else {"position": 0.0, "along_px": 0.0, "across_px": 0.0}
        failed = [
            ("no_label", stats["cover"] < lb["min_cover"]),
            ("label_partial", stats["fill"] < lb["min_fill"]),
            ("label_blurry", stats["sharpness"] < lb["min_sharpness"]),
            ("bottle_hidden", ax["length"] < lb["min_length_ratio"] * ax["body_width"]),
            ("no_label", main is None),
            ("label_off_body", geo["position"] < lb["min_position"]),
            ("label_too_small", min(geo["along_px"], geo["across_px"]) < lb["min_label_px"]),
        ]
        reason = next((name for name, bad in failed if bad), "ok")
        return reason, main if reason == "ok" else None

    def refine_label(self, img: Image.Image, cand: dict, ax: dict, scale: float, label: list) -> list:
        """Если в первом проходе бутылка была мелкой, маска этикетки уточняется вторым проходом SAM3 по вырезке бутылки.

        Если второй проход этикетку не нашёл, остаётся маска первого: бутылка уже прошла проверку.
        """
        lc = self.cfg["label_crop"]
        side = np.ptp(all_points(cand["polys"]), axis=0).max() * (1 + 2 * lc["refine_margin"])
        if min(1.0, self.seg.imgsz / max(side, 1)) / scale < lc["refine_min_gain"]:
            return label
        return self.largest_label(self.seg.labels_on_crop(img, cand["polys"], lc["refine_margin"]), ax) or label

    def _annotate_selected(self, img: Image.Image, cand: dict, scale: float) -> dict:
        """Поля отобранной бутылки: режим кропа, ось, этикетка. Ключ status = "too_small" и т. п., если бутылка не годится."""
        ax = bottle_axis(cand["polys"], self.cfg["orientation"])
        label_status, label = self.check_label(cand, ax) if "label" in cand else ("unchecked", None)
        if self.cfg["label"]["required"] and label_status not in ("ok", "unchecked"):
            return {"status": label_status}
        # маска ненадёжна (бутылка короче min_aspect своих ширин): bbox как есть, без поворота и заглушения фона
        mode = "bbox" if ax["length"] < self.cfg["fallback"]["min_aspect"] * ax["body_width"] else "bottle"
        angle = ax["angle"] if mode == "bottle" else 0.0
        _, (_, _, bw, bh) = rotated_extent(cand["polys"], ax["center"], angle)
        if bh < self.cfg["selection"]["min_bottle_px"]:
            return {"status": "too_small"}
        if label:
            label = self.refine_label(img, cand, ax, scale, label)
        return {
            "status": "selected",
            "mode": mode,
            "angle_deg": round(angle, 2),
            "neck_method": ax["method"],
            "neck_profile_contrast": round(ax["contrast"], 3),
            "neck_point": [round(v, 1) for v in ax["neck"]],
            "base_point": [round(v, 1) for v in ax["base"]],
            "bottle_size_px": [round(float(bw), 1), round(float(bh), 1)],
            "label_status": label_status,
            "label": cand.get("label"),
            "label_polys": label,
        }

    def annotate_image(self, img: Image.Image) -> dict:
        """Разметка без рендера: все кандидаты сегментации с масками и статусом отбора, у отобранных — ось бутылки и этикетка.

        Полигоны в пикселях оригинала после EXIF-поворота. Кандидаты отсортированы по скору, отобранные нумеруются index с 1.
        Этого достаточно, чтобы даталоадер рендерил кроп через render_bottle с джиттером угла, окна, запаса и фона.
        У отобранных: mode ("bottle" или "bbox"), label_status ("ok", причина или "unchecked"), label — метрики этикетки,
        label_polys — маска главной этикетки при label_status == "ok". При label.required бутылка без хорошей этикетки
        не отбирается, причина идёт в status.
        """
        seg = self.seg(img)
        scores = score_candidates(gray_small(img, self.calib["feat_side"]), seg, self.calib)
        max_bottles = self.cfg["selection"]["max_bottles"]
        cands, n_selected = [], 0
        for i in sorted(range(len(scores)), key=lambda i: -scores[i]):
            c = seg["cands"][i]
            entry = {"candidate": c["id"], "conf": c["conf"], "box": c["box"], "score": round(scores[i], 4), "polys": c["polys"]}
            cands.append(entry)
            if scores[i] < self.threshold:
                entry["status"] = "below_threshold"
            elif not any(len(p) >= 3 for p in c["polys"]):
                entry["status"] = "empty_mask"
            elif max_bottles and n_selected >= max_bottles:
                entry["status"] = "over_max_bottles"
            else:
                entry.update(self._annotate_selected(img, c, seg["scale"]))
            if entry["status"] == "selected":
                n_selected += 1
                entry["index"] = n_selected
        return {
            "status": "ok" if n_selected else "no_bottles",
            "width": seg["w"],
            "height": seg["h"],
            "segmentation": {
                "model": seg["model"],
                "prompt": seg["prompt"],
                "imgsz": seg["imgsz"],
                "scale": seg["scale"],
                "n_candidates": len(seg["cands"]),
            },
            "threshold": round(self.threshold, 4),
            "candidates": cands,
        }

    def annotate(self, path: Path | str) -> dict:
        """Разметка одной картинки по пути; см. annotate_image."""
        return self.annotate_image(open_image(path))

    def render(self, rgb: np.ndarray, ann: dict) -> list[Crop]:
        """Кропы отобранных бутылок по разметке, в порядке убывания скора.

        Если ни одна бутылка не отобрана и включён fallback.full_image, возвращается один кроп — целое фото.
        """
        crops = []
        for entry in ann["candidates"]:
            if entry["status"] != "selected":
                continue
            bbox_mode = entry["mode"] == "bbox"
            cfg = {**self.cfg, "background": {**self.cfg["background"], "mode": "none"}} if bbox_mode else self.cfg
            others = [(e["candidate"], e["polys"]) for e in ann["candidates"] if e is not entry and any(len(p) >= 3 for p in e["polys"])]
            crop, info = render_bottle(rgb, entry["polys"], cfg, angle=0.0 if bbox_mode else None, others=others)
            crops.append(Crop(crop, entry["index"], entry["score"], entry["box"], info, entry["mode"], entry["label_status"], entry["label_polys"]))
        if not crops and self.cfg["fallback"]["full_image"]:
            h, w = rgb.shape[:2]
            info = {"angle_deg": 0.0, "matrix_src_to_dst": IDENTITY, "matrix_dst_to_src": IDENTITY, "width": w, "height": h}
            crops.append(Crop(rgb, 1, None, [0, 0, w - 1, h - 1], info, "full_image", "no_bottle"))
        return crops

    def __call__(self, img: Image.Image | Path | str) -> list[Crop]:
        """Картинка → кропы бутылок; при fallback.full_image и отсутствии бутылок — целое фото."""
        if not isinstance(img, Image.Image):
            img = open_image(img)
        return self.render(np.asarray(img), self.annotate_image(img))


def remove_stale(out_dir: Path, stem: str) -> None:
    """Удаляет результаты прошлого прогона этой же картинки: бутылок могло стать меньше."""
    stale = re.compile(re.escape(stem) + r"_b\d+(\.(jpg|png|webp|json)|_label\.(npz|png))")
    for f in out_dir.iterdir():
        if stale.fullmatch(f.name):
            f.unlink()


def write_outputs(path: Path, out_dir: Path, rgb: np.ndarray, crops: list[Crop], cfg: dict) -> list[str]:
    """CLI: на каждую бутылку <имя>_bN.<формат> и <имя>_bN.json; при хорошей этикетке ещё _label.npz и _label.png."""
    fmt, quality = cfg["output"]["format"], cfg["output"]["quality"]
    remove_stale(out_dir, path.stem)
    names = []
    for c in crops:
        stem = f"{path.stem}_b{c.index}"
        save_image(out_dir / f"{stem}.{fmt}", c.image, fmt, quality)
        meta = {
            "source": str(path),
            "mode": c.mode,
            "bottle_box": c.box,
            "bottle_score": c.score,
            "bottle_angle": c.info["angle_deg"],
            "bottle_crop": f"{stem}.{fmt}",
            "bottle_crop_matrix": c.info["matrix_src_to_dst"],
            "label_status": c.label_status,
            "label_box": None,
            "label_mask": None,
            "label_crop": None,
        }
        if c.label_polys:
            label_box = polys_box(c.label_polys)
            np.savez_compressed(out_dir / f"{stem}_label.npz", mask=polys_mask(c.label_polys, label_box))
            save_image(out_dir / f"{stem}_label.png", render_label(rgb, c.label_polys, c.info["angle_deg"], cfg["label_crop"]["padding"]), "png", 0)
            meta.update(label_box=label_box, label_mask=f"{stem}_label.npz", label_crop=f"{stem}_label.png")
        (out_dir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        names.append(f"{stem}.{fmt}")
    return names


def check(cond: bool, *context: object) -> None:
    """Проверка selftest: падает с контекстом вместо assert, который отключается флагом -O."""
    if not cond:
        raise RuntimeError(f"selftest failed: {context}")


def selftest() -> None:
    """Синтетическая бутылка под разными углами: горлышко должно оказаться сверху, запас соблюдён, матрицы обратимы."""
    cfg = tomllib.loads((HERE / "normalize.toml").read_text())
    cfg["background"]["mode"] = "none"
    cfg["orientation"]["upright_within_deg"] = (
        0.0  # сначала проверяем сам профиль ширины на всех углах
    )
    outline = np.array(
        [
            (-7, -60),
            (7, -60),
            (7, -5),
            (20, 10),
            (20, 120),
            (-20, 120),
            (-20, 10),
            (-7, -5),
        ],
        float,
    )
    rgb = np.random.default_rng(0).integers(0, 255, (500, 600, 3), dtype=np.uint8)
    for deg in (0, 30, 90, 135, 180, 250, -45):
        r = math.radians(deg)
        R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
        poly = (outline @ R.T + [300, 250]).round().astype(int).tolist()
        crop, info = render_bottle(rgb, [poly], cfg)
        A = np.array(info["matrix_src_to_dst"])
        Ainv = np.array(info["matrix_dst_to_src"])
        neck_dst = affine(A, np.array([info["neck_point"]]))[0]
        base_dst = affine(A, np.array([info["base_point"]]))[0]
        check(neck_dst[1] < base_dst[1], deg, neck_dst, base_dst)  # горлышко выше дна
        check(abs(neck_dst[0] - base_dst[0]) < 1.5, deg, neck_dst, base_dst)  # строго вертикально
        check(info["neck_method"] == "profile", deg, info["neck_method"])
        back = affine(Ainv, affine(A, np.array([[10.0, 20.0]])))[0]
        check(bool(np.allclose(back, [10, 20], atol=1e-3)), back)  # матрицы взаимно обратны
        bw, bh = info["bottle_size_px"]
        check(abs(crop.shape[0] - bh * 1.2) <= 2 and abs(crop.shape[1] - bw * 1.2) <= 2, deg, crop.shape, bw, bh)
        check(115 <= bh <= 185 and 35 <= bw <= 45, deg, bw, bh)  # вертикальная бутылка 40x180
    cfg["orientation"]["upright_within_deg"] = 30.0
    for deg, method in (
        (10, "upright_prior"),
        (170, "upright_prior"),
        (60, "profile"),
        (135, "profile"),
    ):
        r = math.radians(deg)
        R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
        _, info = render_bottle(
            rgb, [(outline @ R.T + [300, 250]).round().astype(int).tolist()], cfg
        )
        check(info["neck_method"] == method, deg, info["neck_method"])
        if deg == 170:  # почти вертикальная бутылка вверх ногами считается стоящей: верхний конец становится горлышком
            check(abs(info["angle_deg"]) < 15, info["angle_deg"])
    selftest_padding(cfg, rgb, outline)
    selftest_labels(cfg, rgb, outline)
    print("selftest ok")


def selftest_padding(cfg: dict, rgb: np.ndarray, outline: np.ndarray) -> None:
    """Запас не заходит в соседние боксы: бутылка 40x180 с центром (120, 250), её бокс по x 100..140."""
    me = [(outline + np.array([120, 250])).round().astype(int).tolist()]

    def rect(x1: int, x2: int) -> list:
        return [[x1, 160], [x2, 160], [x2, 340], [x1, 340]]

    _, info = render_bottle(rgb, me, cfg, others=[(2, [rect(143, 183)])])  # сосед в 3 px справа
    check(abs(info["padding_px"][0] - 3) < 0.5 and info["padding_limited_by"] == [2], info)  # слева урезано симметрично
    check(abs(info["padding_px"][1] - 18.0) < 0.5, info)  # по вертикали запас полный
    _, info = render_bottle(rgb, me, cfg, others=[(3, [rect(400, 440)])])  # сосед далеко
    check(abs(info["padding_px"][0] - 4) < 0.5 and info["padding_limited_by"] == [], info)
    _, info = render_bottle(rgb, me, cfg, others=[(4, [rect(130, 170)])])  # сосед внахлёст не учитывается
    check(abs(info["padding_px"][0] - 4) < 0.5 and info["padding_limited_by"] == [], info)


def selftest_labels(cfg: dict, rgb: np.ndarray, outline: np.ndarray) -> None:
    """Система координат бутылки, вырезка и маска этикетки, метрики видимости этикетки."""
    r = math.radians(30)
    R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])

    def to_img(pts: list | np.ndarray) -> list:
        return (np.array(pts, float) @ R.T + [300, 250]).round().astype(int).tolist()

    body = [(-20, 40), (20, 40), (20, 80), (-20, 80)]
    ax = bottle_axis([to_img(outline)], cfg["orientation"])  # бутылка 40x180: горлышко y -60..-5, корпус до 120
    check(abs(ax["length"] - 180) < 6 and abs(ax["body_width"] - 40) < 4, ax["length"], ax["body_width"])
    body_label = label_geometry(ax, [to_img(body)])
    neck_label = label_geometry(ax, [to_img([(-7, -50), (7, -50), (7, -30), (-7, -30)])])
    check(body_label["position"] > 0.5 and neck_label["position"] < 0.3, body_label, neck_label)
    check(abs(body_label["along_px"] - 40) < 3 and abs(body_label["width_frac"] - 1) < 0.15, body_label)
    stub = label_geometry({**ax, "length": 60.0}, [to_img(body)])  # видно мало: длина берётся как 3.3 ширины
    check(abs(stub["position"] - body_label["position"] * ax["length"] / (3.3 * ax["body_width"])) < 0.01, stub)

    lcrop = render_label(rgb, [to_img([(-30, 40), (30, 40), (30, 60), (-30, 60)])], ax["angle"], 0.03)
    check(abs(lcrop.shape[1] - 60 * 1.06) <= 3 and abs(lcrop.shape[0] - 20 * 1.06) <= 3, lcrop.shape)  # снова горизонтальна
    square = [[[10, 20], [19, 20], [19, 29], [10, 29]]]
    box = polys_box(square)
    mask = polys_mask(square, box)
    check(box == [10, 20, 19, 29] and mask.shape == (10, 10) and bool(mask.all()), box, mask.shape)

    bottle = np.zeros((100, 40), bool)
    bottle[10:90, 10:30] = True  # бутылка 20x80 = 1600 px
    front, outside, side, cap = (np.zeros_like(bottle) for _ in range(4))
    front[50:70, 10:30] = True  # этикетка 400 px внутри
    outside[0:20, 0:40] = True  # в основном снаружи бутылки
    side[50:70, 26:30] = True  # узкая полоска сбоку: повёрнута
    cap[10:20, 10:30] = True  # «этикетка» на крышке
    st = label_stats(bottle, [0.9, 0.95], [front, outside], 0.4)
    check((st["conf"], st["cover"], st["fill"]) == (0.9, 0.25, 1.0), st)  # внешняя не считается, полоса закрыта
    check(label_stats(bottle, [0.3], [front], 0.4)["cover"] == 0.0)  # слабая не покрывает
    check(label_stats(bottle, [], [], 0.4)["cover"] == 0.0)
    check(label_stats(bottle, [0.9], [side], 0.4)["fill"] == 0.2)
    check(label_stats(bottle, [0.9], [cap], 0.4, exclude=cap)["cover"] == 0.0)
    big = np.zeros((1000, 400), bool)
    big[100:900, 100:300] = True  # резкость на крупной бутылке 200x800
    big_label = np.zeros_like(big)
    big_label[450:650, 100:300] = True
    noisy = np.full((1000, 400), 128, np.uint8)
    noisy[450:650, 100:300] = np.random.default_rng(1).integers(0, 255, (200, 200))
    check(label_stats(big, [0.9], [big_label], 0.4, gray_full=noisy)["sharpness"] > 300)  # мелкий контрастный рисунок
    check(label_stats(big, [0.9], [big_label], 0.4, gray_full=cv2.GaussianBlur(noisy, (0, 0), 6))["sharpness"] < 300)  # расфокус
    check(label_stats(bottle, [0.9], [front], 0.4, gray_full=noisy[:100, :40])["sharpness"] < 300)  # 20 px: не читается


@click.command()
@click.argument("inputs", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("-c", "--config", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=HERE / "normalize.toml", show_default=True)
@click.option("--set", "overrides", multiple=True, metavar="секция.ключ=значение", help="Переопределить параметр конфига")
@click.option("--selftest", "selftest_", is_flag=True, help="Проверка геометрии на синтетике, без модели")
def main(inputs: tuple[Path, ...], config: Path, overrides: tuple[str, ...], selftest_: bool) -> None:
    """Нормализация бутылок вина: SAM3 → отбор по калибровке → поворот горлышком вверх → кроп с запасом → блюр фона.

    INPUTS — картинки или папки. На каждую бутылку в output.dir пишутся <имя>_bN.<формат> и <имя>_bN.json,
    при хорошей этикетке ещё <имя>_bN_label.npz (маска) и <имя>_bN_label.png (вырезка под OCR).
    """
    if selftest_:
        selftest()
        return
    if not inputs:
        raise click.UsageError("нужна хотя бы одна картинка или папка")
    cfg = load_config(config, list(overrides))
    base = config.resolve().parent
    files = [f for p in inputs for f in (sorted(p.iterdir()) if p.is_dir() else [p]) if f.suffix.lower() in EXTS]
    out_dir = resolve(cfg["output"]["dir"], base)
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = [f.stem for f in files]
    if len(set(stems)) < len(stems):  # выходные файлы называются по имени без расширения
        raise click.UsageError(f"одинаковые имена без расширения: {sorted({s for s in stems if stems.count(s) > 1})[:5]}")
    if any(f.resolve().parent == out_dir.resolve() for f in files):
        raise click.UsageError("output.dir совпадает с папкой исходников: очистка старых результатов могла бы удалить исходные картинки")
    norm = Normalizer(cfg, base)
    for f in files:
        try:
            t0 = time.perf_counter()
            img = open_image(f)
            ann = norm.annotate_image(img)
            rgb = np.asarray(img)
            crops = norm.render(rgb, ann)
            write_outputs(f, out_dir, rgb, crops, cfg)
            kinds = ", ".join(f"{c.mode}/{c.label_status}" for c in crops)
            click.echo(
                f"{f.name}: {ann['status']}, кропов {len(crops)} из {ann['segmentation']['n_candidates']} кандидатов"
                f"{f' ({kinds})' if kinds else ''}, {round((time.perf_counter() - t0) * 1000)} мс"
            )
        except Exception as e:  # одна битая картинка не должна ронять весь пакет
            click.echo(f"{f.name}: ошибка {e!r}", err=True)


if __name__ == "__main__":
    main()
