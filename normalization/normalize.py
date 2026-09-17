#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["ultralytics", "pillow", "numpy", "timm", "clip @ git+https://github.com/ultralytics/CLIP.git"]
# ///
"""
Нормализация бутылок вина: SAM3 → отбор по калибровке → проверка этикетки → поворот горлышком вверх,
кроп с запасом и блюр фона → маска и вырезка этикетки.

  normalize.py <картинка или папка> [...] [-c normalize.toml] [--set секция.ключ=значение ...]
  normalize.py --selftest

На каждую вырезанную бутылку пишутся <имя>_bN.<формат>, <имя>_bN.json, <имя>_bN_label.npz и <имя>_bN_label.png.
Формат описан в README.md.
"""
import argparse, json, math, re, sys, time, tomllib
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BORDERS = {"replicate": cv2.BORDER_REPLICATE, "reflect": cv2.BORDER_REFLECT_101, "constant": cv2.BORDER_CONSTANT}


# ---------- конфиг ----------

def load_config(path: Path, overrides: list[str]) -> dict:
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
    return Path(p) if Path(p).is_absolute() else base / p


# ---------- отбор бутылок ----------

def score_candidates(gray_small: np.ndarray, seg: dict, calib: dict) -> list[float]:
    from features import candidate_features
    feats = candidate_features(gray_small, seg)
    if not feats:
        return []
    X = np.array([[f[k] for k in calib["features"]] for f in feats])
    z = (X - np.array(calib["mean"])) / np.array(calib["std"])
    logit = z @ np.array(calib["coef"]) + calib["intercept"]
    return (1 / (1 + np.exp(-logit))).tolist()


# ---------- геометрия ----------

def all_points(polys: list) -> np.ndarray:
    return np.concatenate([np.asarray(p, np.float64) for p in polys])


def bottle_axis(polys: list, orient: dict) -> dict:
    """Ось бутылки по маске: точки горлышка и дна, угол поворота до вертикали горлышком вверх,
    единичные векторы вдоль (от горлышка к дну) и поперёк оси, видимая длина и ширина корпуса."""
    allp = all_points(polys)
    (x1, y1), (x2, y2) = allp.min(0), allp.max(0)
    s = 400 / max(x2 - x1, y2 - y1, 1)  # считаем на маске ~400 px, этого хватает
    m = np.zeros((int((y2 - y1) * s) + 3, int((x2 - x1) * s) + 3), np.uint8)
    cv2.fillPoly(m, [np.round((np.asarray(p, np.float64) - [x1, y1]) * s).astype(np.int32) + 1 for p in polys], 1)
    ys, xs = np.nonzero(m)
    pts = (np.stack([xs, ys], 1).astype(np.float64) - 1) / s + [x1, y1]
    center = pts.mean(0)
    axis = np.linalg.eigh(np.cov((pts - center).T))[1][:, -1]
    t = (pts - center) @ axis
    hist, edges = np.histogram(t, bins=20)
    widths = hist / max(edges[1] - edges[0], 1e-9)
    a_end, b_end = widths[:3].mean(), widths[-3:].mean()  # ширина у концов «минус» и «плюс» оси
    contrast = abs(a_end - b_end) / (np.percentile(widths, 90) + 1e-9)
    p_minus, p_plus = center + axis * t.min(), center + axis * t.max()
    tilt = math.degrees(math.acos(min(1.0, abs(axis[1]))))  # отклонение оси от вертикали, 0..90
    upper = (p_minus, p_plus) if p_minus[1] < p_plus[1] else (p_plus, p_minus)
    if tilt <= orient["upright_within_deg"]:
        # профиль ширины путает дно с горлышком при дырах и бликах в маске; почти вертикальную бутылку считаем стоящей
        neck, base, method = *upper, "upright_prior"
    elif orient["neck"] == "profile" and contrast >= orient["min_profile_contrast"]:
        neck, base, method = (p_minus, p_plus, "profile") if a_end < b_end else (p_plus, p_minus, "profile")
    else:
        neck, base, method = *upper, "up"
    length = float(np.ptp(t))
    u = (base - neck) / max(length, 1e-9)
    n = np.array([-u[1], u[0]])
    # cv2.getRotationMatrix2D(angle=a+90) переводит направление с углом a (ось y вниз) в (0, -1), то есть вверх
    angle = math.degrees(math.atan2(-u[1], -u[0])) + 90 if orient["enabled"] else 0.0
    # ширина корпуса: 90-й перцентиль размаха маски поперёк оси по 30 срезам
    w = (pts - neck) @ n
    bins = np.clip(((pts - neck) @ u / max(length, 1e-9) * 30).astype(int), 0, 29)
    spans = [np.ptp(w[bins == i]) if (bins == i).sum() > 3 else 0.0 for i in range(30)]
    return {"center": center, "neck": neck, "base": base, "angle": angle, "method": method,
            "u": u, "n": n, "length": length, "body_width": float(np.percentile(spans, 90))}


def label_geometry(ax: dict, polys: list) -> dict:
    """Положение этикетки вдоль бутылки (доля полной длины от горлышка) и её размеры в системе бутылки."""
    lp = all_points(polys) - ax["neck"]
    full = max(ax["length"], 3.3 * ax["body_width"])  # корпус может быть скрыт: полная длина не меньше 3.3 ширины
    t, w = lp @ ax["u"], lp @ ax["n"]
    return {"position": float(t.mean() / max(full, 1e-9)), "along_px": float(np.ptp(t)), "across_px": float(np.ptp(w)),
            "width_frac": float(np.ptp(w) / max(ax["body_width"], 1e-9))}


def affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ M[:, :2].T + M[:, 2]


def fast_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    # размываем уменьшенную копию, на больших кропах честный Гаусс с sigma ~100 px идёт секундами
    h, w = img.shape[:2]
    f = max(1.0, sigma / 3)
    small = cv2.resize(img, (max(1, round(w / f)), max(1, round(h / f))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def limit_padding(own, others, px: float, py: float) -> tuple[float, float, list]:
    """Урезает запас, чтобы он не заходил в боксы соседей. Боксы [x1, y1, x2, y2] в системе координат кропа.

    Сосед, перекрывающий бокс самой бутылки, не учитывается: запасом это не исправить.
    Запас по оси урезается симметрично, чтобы бутылка оставалась по центру кропа.
    Возвращает px, py и номера соседей, которые урезали запас.
    """
    x1, y1, x2, y2 = own
    side = {"left": (px, None), "right": (px, None), "top": (py, None), "bottom": (py, None)}

    def tighten(name, gap, oid):
        if gap < side[name][0]:
            side[name] = (max(0.0, gap), oid)

    for oid, (ox1, oy1, ox2, oy2) in others:
        if oy2 > y1 - py and oy1 < y2 + py:  # сосед на уровне кропа по вертикали
            if ox2 <= x1:
                tighten("left", x1 - ox2, oid)
            elif ox1 >= x2:
                tighten("right", ox1 - x2, oid)
    px = min(side["left"][0], side["right"][0])
    for oid, (ox1, oy1, ox2, oy2) in others:
        if ox2 > x1 - px and ox1 < x2 + px:  # сосед на уровне кропа по горизонтали, с уже урезанным запасом
            if oy2 <= y1:
                tighten("top", y1 - oy2, oid)
            elif oy1 >= y2:
                tighten("bottom", oy1 - y2, oid)
    py = min(side["top"][0], side["bottom"][0])
    binding = [side[s][1] for s in ("left", "right") if side[s][1] is not None and side[s][0] == px]
    binding += [side[s][1] for s in ("top", "bottom") if side[s][1] is not None and side[s][0] == py]
    return px, py, sorted(set(binding))


def render_bottle(rgb: np.ndarray, polys: list, cfg: dict, others=(), ax: dict | None = None) -> tuple[np.ndarray, dict]:
    """Кроп одной бутылки: поворот, запас, блюр фона. others: [(id кандидата, полигоны)] соседних бутылок."""
    ax = ax or bottle_axis(polys, cfg["orientation"])
    M = cv2.getRotationMatrix2D(tuple(map(float, ax["center"])), ax["angle"], 1.0)
    verts = affine(M, all_points(polys))
    (rx1, ry1), (rx2, ry2) = verts.min(0), verts.max(0)
    bw, bh = rx2 - rx1, ry2 - ry1
    px, py = cfg["crop"]["padding_x"] * bw, cfg["crop"]["padding_y"] * bh
    limited_by = []
    if cfg["crop"]["avoid_neighbors"] and others:
        boxes = []
        for oid, opolys in others:  # соседей поворачиваем той же матрицей, что и бутылку
            ov = affine(M, all_points(opolys))
            boxes.append((oid, (*ov.min(0), *ov.max(0))))
        px, py, limited_by = limit_padding((rx1, ry1, rx2, ry2), boxes, px, py)
    W, H = max(1, math.ceil(bw + 2 * px)), max(1, math.ceil(bh + 2 * py))
    A = M.copy()
    A[:, 2] -= [rx1 - px, ry1 - py]
    border = BORDERS[cfg["crop"]["border"]]
    crop = cv2.warpAffine(rgb, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=border, borderValue=(0, 0, 0))

    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [np.round(affine(A, np.asarray(p, np.float64))).astype(np.int32) for p in polys], 255)
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
        "angle_deg": round(ax["angle"], 2),
        "neck_method": ax["method"],
        "neck_point": [round(v, 1) for v in ax["neck"]],
        "base_point": [round(v, 1) for v in ax["base"]],
        "bottle_size_px": [round(bw, 1), round(bh, 1)],
        "padding_px": [round(px, 1), round(py, 1)],
        "padding_limited_by": limited_by,
        "matrix_src_to_dst": np.round(A, 6).tolist(),
    }
    return crop, info


def polys_box(polys: list) -> list:
    (x1, y1), (x2, y2) = all_points(polys).min(0), all_points(polys).max(0)
    return [int(x1), int(y1), int(x2), int(y2)]


def polys_mask(polys: list, box: list) -> np.ndarray:
    """bool-маска размером с бокс [x1, y1, x2, y2] включительно: mask[y - y1, x - x1] для пикселя оригинала (x, y)."""
    x1, y1, x2, y2 = box
    m = np.zeros((y2 - y1 + 1, x2 - x1 + 1), np.uint8)
    cv2.fillPoly(m, [np.asarray(p, np.int32) - [x1, y1] for p in polys], 1)
    return m.astype(bool)


def render_label(rgb: np.ndarray, polys: list, angle: float, padding: float) -> np.ndarray:
    """Вырезка этикетки под OCR: оригинал в полном разрешении, поворот как у бутылки (текст горизонтально),
    прямоугольник вокруг маски с запасом. Без блюра и маскирования, чтобы не срезать края букв."""
    pts = all_points(polys)
    M = cv2.getRotationMatrix2D(tuple(map(float, pts.mean(0))), angle, 1.0)
    v = affine(M, pts)
    (x1, y1), (x2, y2) = v.min(0), v.max(0)
    px, py = padding * (x2 - x1), padding * (y2 - y1)
    W, H = max(1, math.ceil(x2 - x1 + 2 * px)), max(1, math.ceil(y2 - y1 + 2 * py))
    A = M.copy()
    A[:, 2] -= [x1 - px, y1 - py]
    return cv2.warpAffine(rgb, A, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def save_image(path: Path, rgb: np.ndarray, fmt: str, quality: int):
    params = {"jpg": [cv2.IMWRITE_JPEG_QUALITY, quality], "webp": [cv2.IMWRITE_WEBP_QUALITY, quality], "png": []}[fmt]
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), params):
        raise OSError(f"не удалось записать {path}")


# ---------- прогон ----------

class Normalizer:
    def __init__(self, cfg: dict, base: Path):
        from seg import Segmenter
        self.cfg = cfg
        sel, sc, lb = cfg["selection"], cfg["segmentation"], cfg["label"]
        if not lb["prompt"]:
            raise SystemExit("label.prompt пуст: без этикетки бутылки не вырезаются, проверку отключить нельзя")
        self.calib = json.loads(resolve(sel["calibration"], base).read_text())
        self.threshold = sel["threshold"] if sel["threshold"] >= 0 else self.calib["threshold"]
        self.seg = Segmenter(resolve(sc["model"], base), sc["prompt"], sc["conf"], sc["imgsz"], sc["max_side"],
                             label_prompt=lb["prompt"], label_conf=lb["min_conf"], exclude_prompts=lb["exclude_prompts"])
        if self.seg.tag != self.calib.get("segmentation"):
            print(f"внимание: калибровка сделана для сегментации {self.calib.get('segmentation')}, сейчас {self.seg.tag}", file=sys.stderr)

    def check_label(self, cand: dict, ax: dict) -> tuple[str | None, list | None]:
        """(причина отказа, None) или (None, полигоны главной этикетки корпуса из первого прохода).

        Порядок: есть ли этикетка, видна ли, резкая ли, виден ли корпус, где этикетка и какого она размера.
        Главная этикетка: самая крупная на корпусе, полоски уже min_width_frac ширины бутылки (акцизные марки) не считаются.
        """
        lb, stats = self.cfg["label"], cand["label"]
        if stats["cover"] < lb["min_cover"]:
            return "no_label", None
        if stats["fill"] < lb["min_fill"]:
            return "label_partial", None
        if stats["sharpness"] < lb["min_sharpness"]:
            return "label_blurry", None
        if ax["length"] < lb["min_length_ratio"] * ax["body_width"]:
            return "bottle_hidden", None
        main = self.largest_label(cand["label_polys"], ax)
        if main is None:
            return "no_label", None
        geo = label_geometry(ax, main)
        if geo["position"] < lb["min_position"]:
            return "label_off_body", None
        if min(geo["along_px"], geo["across_px"]) < lb["min_label_px"]:
            return "label_too_small", None
        return None, main

    def largest_label(self, labels: list, ax: dict) -> list | None:
        polys = [l["polys"] for l in labels
                 if l["polys"] and label_geometry(ax, l["polys"])["width_frac"] >= self.cfg["label_crop"]["min_width_frac"]]
        return max(polys, key=lambda ps: sum(cv2.contourArea(np.asarray(p, np.float32)) for p in ps), default=None)

    def refine_label(self, img, cand: dict, ax: dict, scale: float, label: list) -> list:
        """Если в первом проходе бутылка была мелкой, маска этикетки уточняется вторым проходом SAM3 по вырезке бутылки.
        Если второй проход этикетку не нашёл, остаётся маска первого: бутылка уже прошла проверку."""
        lc = self.cfg["label_crop"]
        side = np.ptp(all_points(cand["polys"]), axis=0).max() * (1 + 2 * lc["refine_margin"])
        if min(1.0, self.seg.imgsz / max(side, 1)) / scale < lc["refine_min_gain"]:
            return label
        return self.largest_label(self.seg.labels_on_crop(img, cand["polys"], lc["refine_margin"]), ax) or label

    def __call__(self, path: Path, out_dir: Path) -> dict:
        """Пишет файлы на каждую вырезанную бутылку. Возвращает их JSON и счётчик причин отказа."""
        from features import gray_small
        from seg import open_image
        img = open_image(path)
        seg = self.seg(img)
        rgb = np.asarray(img)
        scores = score_candidates(gray_small(img, self.calib["feat_side"]), seg, self.calib)
        sel, out = self.cfg["selection"], self.cfg["output"]
        stale = re.compile(re.escape(path.stem) + r"_b\d+(\.(jpg|png|webp|json)|_label\.(npz|png))")
        for f in out_dir.iterdir():  # бутылки прошлого прогона этой же картинки: их может стать меньше
            if stale.fullmatch(f.name):
                f.unlink()
        bottles, rejected = [], {}
        for i in sorted(range(len(scores)), key=lambda i: -scores[i]):
            c = seg["cands"][i]
            label = ax = label_status = None
            if scores[i] < self.threshold or not c["polys"]:
                reason = "below_threshold"
            elif sel["max_bottles"] and len(bottles) >= sel["max_bottles"]:
                reason = "over_max_bottles"
            else:
                ax = bottle_axis(c["polys"], self.cfg["orientation"])
                label_status, label = self.check_label(c, ax)
                reason = label_status if self.cfg["label"]["required"] else None
            if not reason:
                mode = "bottle"
                if ax["length"] < self.cfg["fallback"]["min_aspect"] * ax["body_width"]:
                    # маска ненадёжна: bbox как есть, без поворота и заглушения фона
                    mode, ax = "bbox", {**ax, "angle": 0.0}
                cfg = self.cfg if mode == "bottle" else {**self.cfg, "background": {**self.cfg["background"], "mode": "none"}}
                others = [(o["id"], o["polys"]) for o in seg["cands"] if o is not c and o["polys"]]
                crop, info = render_bottle(rgb, c["polys"], cfg, others, ax)
                if info["bottle_size_px"][1] < sel["min_bottle_px"]:
                    reason = "too_small"
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            if label:
                label = self.refine_label(img, c, ax, seg["scale"], label)
            bottles.append(self.write(path, out_dir, len(bottles) + 1, rgb, crop, mode, c["box"], round(scores[i], 4),
                                      info["angle_deg"], info["matrix_src_to_dst"], label_status or "ok", label))
        if not bottles and self.cfg["fallback"]["full_image"]:
            # детектор бутылку не нашёл: в поиск идёт целое фото
            h, w = rgb.shape[:2]
            bottles.append(self.write(path, out_dir, 1, rgb, rgb, "full_image", [0, 0, w - 1, h - 1], None, 0.0,
                                      [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "no_bottle", None))
        return {"bottles": bottles, "rejected": rejected}

    def write(self, path: Path, out_dir: Path, n: int, rgb, crop, mode: str, box: list, score, angle: float,
              matrix: list, label_status: str, label: list | None) -> dict:
        """Файлы одной бутылки. Маска и вырезка этикетки пишутся, только если этикетка прошла проверку."""
        out, stem = self.cfg["output"], f"{path.stem}_b{n}"
        save_image(out_dir / f"{stem}.{out['format']}", crop, out["format"], out["quality"])
        meta = {
            "source": str(path),
            "mode": mode,
            "bottle_box": box,
            "bottle_score": score,
            "bottle_angle": angle,
            "bottle_crop": f"{stem}.{out['format']}",
            "bottle_crop_matrix": matrix,
            "label_status": label_status,
            "label_box": None, "label_mask": None, "label_crop": None,
        }
        if label_status == "ok":
            label_box = polys_box(label)
            np.savez_compressed(out_dir / f"{stem}_label.npz", mask=polys_mask(label, label_box))
            save_image(out_dir / f"{stem}_label.png", render_label(rgb, label, angle, self.cfg["label_crop"]["padding"]), "png", 0)
            meta.update(label_box=label_box, label_mask=f"{stem}_label.npz", label_crop=f"{stem}_label.png")
        (out_dir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        return meta


def selftest():
    """Синтетическая бутылка под разными углами: горлышко должно оказаться сверху, запас соблюдён, матрицы обратимы."""
    cfg = tomllib.loads((HERE / "normalize.toml").read_text())
    cfg["background"]["mode"] = "none"
    cfg["orientation"]["upright_within_deg"] = 0.0  # сначала проверяем сам профиль ширины на всех углах
    outline = np.array([(-7, -60), (7, -60), (7, -5), (20, 10), (20, 120), (-20, 120), (-20, 10), (-7, -5)], float)
    rgb = np.random.default_rng(0).integers(0, 255, (500, 600, 3), dtype=np.uint8)
    for deg in (0, 30, 90, 135, 180, 250, -45):
        r = math.radians(deg)
        R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
        poly = (outline @ R.T + [300, 250]).round().astype(int).tolist()
        crop, info = render_bottle(rgb, [poly], cfg)
        A = np.array(info["matrix_src_to_dst"])
        neck_dst = affine(A, np.array([info["neck_point"]]))[0]
        base_dst = affine(A, np.array([info["base_point"]]))[0]
        assert neck_dst[1] < base_dst[1], (deg, neck_dst, base_dst)                   # горлышко выше дна
        assert abs(neck_dst[0] - base_dst[0]) < 1.5, (deg, neck_dst, base_dst)        # строго вертикально
        assert info["neck_method"] == "profile", (deg, info["neck_method"])
        bw, bh = info["bottle_size_px"]
        assert abs(crop.shape[0] - bh * 1.2) <= 2 and abs(crop.shape[1] - bw * 1.2) <= 2, (deg, crop.shape, bw, bh)
        assert 115 <= bh <= 185 and 35 <= bw <= 45, (deg, bw, bh)                     # вертикальная бутылка 40x180
    cfg["orientation"]["upright_within_deg"] = 30.0
    for deg, method in ((10, "upright_prior"), (170, "upright_prior"), (60, "profile"), (135, "profile")):
        r = math.radians(deg)
        R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
        _, info = render_bottle(rgb, [(outline @ R.T + [300, 250]).round().astype(int).tolist()], cfg)
        assert info["neck_method"] == method, (deg, info["neck_method"])
        if deg == 170:  # почти вертикальная бутылка вверх ногами считается стоящей: верхний конец становится горлышком
            assert abs(info["angle_deg"]) < 15, info["angle_deg"]
    # запас не заходит в соседние боксы: бутылка 40x180 с центром (120, 250), её бокс x 100..140
    cfg["crop"]["avoid_neighbors"] = True
    me = [(outline + [120, 250]).round().astype(int).tolist()]
    rect = lambda x1, x2: [[x1, 160], [x2, 160], [x2, 340], [x1, 340]]
    _, info = render_bottle(rgb, me, cfg, [(2, [rect(143, 183)])])                       # сосед в 3 px справа
    assert abs(info["padding_px"][0] - 3) < 0.5 and info["padding_limited_by"] == [2], info       # слева урезано симметрично
    assert abs(info["padding_px"][1] - 18.0) < 0.5, info                                   # по вертикали запас полный
    _, info = render_bottle(rgb, me, cfg, [(3, [rect(400, 440)])])                        # сосед далеко
    assert abs(info["padding_px"][0] - 4) < 0.5 and info["padding_limited_by"] == [], info
    _, info = render_bottle(rgb, me, cfg, [(4, [rect(130, 170)])])                        # сосед внахлёст не учитывается
    assert abs(info["padding_px"][0] - 4) < 0.5 and info["padding_limited_by"] == [], info

    # система бутылки: синтетическая бутылка 40x180 (горлышко y -60..-5, корпус до 120) под углом 30°
    r = math.radians(30)
    R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    to_img = lambda pts: (np.array(pts, float) @ R.T + [300, 250]).round().astype(int).tolist()
    frame = bottle_axis([to_img(outline)], cfg["orientation"])
    assert abs(frame["length"] - 180) < 6 and abs(frame["body_width"] - 40) < 4, (frame["length"], frame["body_width"])
    body_label = label_geometry(frame, [to_img([(-20, 40), (20, 40), (20, 80), (-20, 80)])])
    neck_label = label_geometry(frame, [to_img([(-7, -50), (7, -50), (7, -30), (-7, -30)])])
    assert body_label["position"] > 0.5 and neck_label["position"] < 0.3, (body_label, neck_label)
    assert abs(body_label["along_px"] - 40) < 3 and abs(body_label["width_frac"] - 1) < 0.15, body_label
    stub = label_geometry({**frame, "length": 60.0}, [to_img([(-20, 40), (20, 40), (20, 80), (-20, 80)])])
    expected = body_label["position"] * frame["length"] / (3.3 * frame["body_width"])   # видно мало: длина берётся как 3.3 ширины
    assert abs(stub["position"] - expected) < 0.01, (stub, expected)

    # вырезка этикетки: этикетка 60x20 на той же бутылке под углом 30°, после поворота снова горизонтальна
    lcrop = render_label(rgb, [to_img([(-30, 40), (30, 40), (30, 60), (-30, 60)])], frame["angle"], 0.03)
    assert abs(lcrop.shape[1] - 60 * 1.06) <= 3 and abs(lcrop.shape[0] - 20 * 1.06) <= 3, lcrop.shape  # горизонтально, с запасом
    square = [[[10, 20], [19, 20], [19, 29], [10, 29]]]                                     # маска этикетки по боксу
    box = polys_box(square)
    mask = polys_mask(square, box)
    assert box == [10, 20, 19, 29] and mask.shape == (10, 10) and mask.all(), (box, mask.shape)

    from seg import label_stats
    bottle = np.zeros((100, 40), bool); bottle[10:90, 10:30] = True                       # бутылка 20x80 = 1600 px
    front = np.zeros_like(bottle); front[50:70, 10:30] = True                               # этикетка 400 px внутри
    outside = np.zeros_like(bottle); outside[0:20, 0:40] = True                             # в основном снаружи бутылки
    st = label_stats(bottle, [0.9, 0.95], [front, outside], 0.4)
    assert (st["conf"], st["cover"], st["fill"]) == (0.9, 0.25, 1.0), st                   # внешняя не считается, полоса закрыта
    assert label_stats(bottle, [0.3], [front], 0.4)["cover"] == 0.0                         # слабая не покрывает
    assert label_stats(bottle, [], [], 0.4)["cover"] == 0.0
    side = np.zeros_like(bottle); side[50:70, 26:30] = True                                 # узкая полоска сбоку: повёрнута
    assert label_stats(bottle, [0.9], [side], 0.4)["fill"] == 0.2
    cap = np.zeros_like(bottle); cap[10:20, 10:30] = True                                   # «этикетка» на крышке
    assert label_stats(bottle, [0.9], [cap], 0.4, exclude=cap)["cover"] == 0.0
    rng = np.random.default_rng(1)                                                          # резкость на крупной бутылке 200x800
    big = np.zeros((1000, 400), bool); big[100:900, 100:300] = True
    big_label = np.zeros_like(big); big_label[450:650, 100:300] = True
    noisy = np.full((1000, 400), 128, np.uint8); noisy[450:650, 100:300] = rng.integers(0, 255, (200, 200))
    assert label_stats(big, [0.9], [big_label], 0.4, gray_full=noisy)["sharpness"] > 300    # мелкий контрастный рисунок
    assert label_stats(big, [0.9], [big_label], 0.4, gray_full=cv2.GaussianBlur(noisy, (0, 0), 6))["sharpness"] < 300  # расфокус
    assert label_stats(bottle, [0.9], [front], 0.4, gray_full=noisy[:100, :40])["sharpness"] < 300  # 20 px: не читается
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", type=Path)
    ap.add_argument("-c", "--config", type=Path, default=HERE / "normalize.toml")
    ap.add_argument("--set", action="append", default=[], metavar="секция.ключ=значение")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(HERE))
    if args.selftest:
        return selftest()
    cfg = load_config(args.config, args.set)
    base = args.config.resolve().parent
    files = [f for p in args.inputs for f in (sorted(p.iterdir()) if p.is_dir() else [p]) if f.suffix.lower() in EXTS]
    if not files:
        ap.error("нужна хотя бы одна картинка или папка с картинками")
    stems = [f.stem for f in files]
    if len(set(stems)) < len(stems):  # выходные файлы называются по имени без расширения
        ap.error(f"одинаковые имена без расширения: {sorted({s for s in stems if stems.count(s) > 1})[:5]}")
    out_dir = resolve(cfg["output"]["dir"], base)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(f.resolve().parent == out_dir.resolve() for f in files):
        ap.error("output.dir совпадает с папкой исходников: очистка старых результатов могла бы удалить исходные картинки")
    norm = Normalizer(cfg, base)
    for f in files:
        try:
            t = time.perf_counter()
            r = norm(f, out_dir)
            why = ", ".join(f"{k} {v}" for k, v in r["rejected"].items() if k != "below_threshold")
            kinds = ", ".join(f"{b['mode']}/{b['label_status']}" for b in r["bottles"])
            print(f"{f.name}: вырезано {len(r['bottles'])}" + (f" ({kinds})" if kinds else "") + (f", отказы: {why}" if why else "") +
                  f", {round((time.perf_counter() - t) * 1000)} мс")
        except Exception as e:  # одна битая картинка не должна ронять весь пакет
            print(f"{f.name}: ошибка {e!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
