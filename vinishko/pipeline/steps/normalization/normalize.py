#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["ultralytics", "pillow", "numpy", "timm", "clip @ git+https://github.com/ultralytics/CLIP.git"]
# ///
"""
Нормализация бутылок вина: SAM3 → отбор по калибровке → поворот горлышком вверх → кроп с запасом → блюр фона.

  normalize.py <картинка или папка> [...] [-c normalize.toml] [--set секция.ключ=значение ...]
  normalize.py --selftest

На каждую картинку пишется <имя>.json, на каждую отобранную бутылку <имя>_b<N>.<формат>.
В JSON есть матрицы src_to_dst и dst_to_src (аффинные 2x3): точка оригинала [x, y, 1] -> точка выходного кропа.
"""
import argparse, hashlib, json, math, sys, time, tomllib
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

def bottle_axis(polys: list, orient: dict) -> dict:
    """Ось бутылки по маске: центр, точки горлышка и дна, угол поворота до вертикали горлышком вверх."""
    arrs = [np.asarray(p, np.float32) for p in polys if len(p) >= 3]
    allp = np.concatenate(arrs)
    x1, y1 = allp.min(0); x2, y2 = allp.max(0)
    s = 256 / max(x2 - x1, y2 - y1, 1)  # профиль считаем на маске ~256 px, этого хватает
    m = np.zeros((int((y2 - y1) * s) + 3, int((x2 - x1) * s) + 3), np.uint8)
    cv2.fillPoly(m, [np.round((a - [x1, y1]) * s).astype(np.int32) + 1 for a in arrs], 1)
    ys, xs = np.nonzero(m)
    pts = (np.stack([xs, ys], 1).astype(np.float64) - 1) / s + [x1, y1]
    center = pts.mean(0)
    axis = np.linalg.eigh(np.cov((pts - center).T))[1][:, -1]
    t = (pts - center) @ axis
    hist, edges = np.histogram(t, bins=20)
    widths = hist / max(edges[1] - edges[0], 1e-9)
    wmax = np.percentile(widths, 90) + 1e-9
    a_end, b_end = widths[:3].mean(), widths[-3:].mean()  # ширина у концов «минус» и «плюс» оси
    contrast = abs(a_end - b_end) / wmax
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
    v = neck - base
    # cv2.getRotationMatrix2D(angle=a+90) переводит направление с углом a (ось y вниз) в (0, -1), то есть вверх
    angle = math.degrees(math.atan2(v[1], v[0])) + 90 if orient["enabled"] else 0.0
    return {"center": center, "neck": neck, "base": base, "angle": angle, "method": method, "contrast": float(contrast)}


def affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ M[:, :2].T + M[:, 2]


def fast_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    # размываем уменьшенную копию, на больших кропах честный Гаусс с sigma ~100 px идёт секундами
    h, w = img.shape[:2]
    f = max(1.0, sigma / 3)
    small = cv2.resize(img, (max(1, round(w / f)), max(1, round(h / f))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def render_bottle(rgb: np.ndarray, polys: list, cfg: dict) -> tuple[np.ndarray, dict]:
    """Кроп одной бутылки: поворот, запас, блюр фона. Возвращает картинку и описание преобразования."""
    ax = bottle_axis(polys, cfg["orientation"])
    M = cv2.getRotationMatrix2D(tuple(map(float, ax["center"])), ax["angle"], 1.0)
    verts = affine(M, np.concatenate([np.asarray(p, np.float64) for p in polys if len(p) >= 3]))
    (rx1, ry1), (rx2, ry2) = verts.min(0), verts.max(0)
    bw, bh = rx2 - rx1, ry2 - ry1
    px, py = cfg["crop"]["padding_x"] * bw, cfg["crop"]["padding_y"] * bh
    W, H = max(1, math.ceil(bw + 2 * px)), max(1, math.ceil(bh + 2 * py))
    A = M.copy()
    A[:, 2] -= [rx1 - px, ry1 - py]
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
        "angle_deg": round(ax["angle"], 2),
        "neck_method": ax["method"],
        "neck_profile_contrast": round(ax["contrast"], 3),
        "neck_point": [round(v, 1) for v in ax["neck"]],
        "base_point": [round(v, 1) for v in ax["base"]],
        "bottle_size_px": [round(bw, 1), round(bh, 1)],
        "padding_px": [round(px, 1), round(py, 1)],
        "out_of_bounds_frac": round(1 - float(inside.mean()), 4),
        "matrix_src_to_dst": np.round(A, 6).tolist(),
        "matrix_dst_to_src": np.round(cv2.invertAffineTransform(A), 6).tolist(),
        "width": crop.shape[1], "height": crop.shape[0],
    }
    return crop, info


# ---------- прогон ----------

class Normalizer:
    def __init__(self, cfg: dict, base: Path):
        self.cfg, self.base = cfg, base
        sel = cfg["selection"]
        self.calib = json.loads(resolve(sel["calibration"], base).read_text())
        self.threshold = sel["threshold"] if sel["threshold"] >= 0 else self.calib["threshold"]
        sc = cfg["segmentation"]
        from seg import Segmenter
        self.seg = Segmenter(resolve(sc["model"], base), sc["prompt"], sc["conf"], sc["imgsz"], sc["max_side"])
        if self.seg.tag != self.calib.get("segmentation"):
            print(f"внимание: калибровка сделана для сегментации {self.calib.get('segmentation')}, сейчас {self.seg.tag}", file=sys.stderr)

    def __call__(self, path: Path, out_dir: Path) -> dict:
        from features import gray_small
        from seg import open_image
        timing, t0 = {}, time.perf_counter()
        img = open_image(path)
        rgb = np.asarray(img)
        timing["decode"] = time.perf_counter() - t0

        t = time.perf_counter()
        seg = self.seg(img)
        timing["segmentation"] = time.perf_counter() - t

        t = time.perf_counter()
        scores = score_candidates(gray_small(img, self.calib["feat_side"]), seg, self.calib)
        timing["selection"] = time.perf_counter() - t

        sel, cfg = self.cfg["selection"], self.cfg
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        cands, bottles = [], []
        t = time.perf_counter()
        for rank, i in enumerate(order):
            c = seg["cands"][i]
            entry = {"candidate": c["id"], "conf": c["conf"], "box": c["box"], "score": round(scores[i], 4)}
            cands.append(entry)
            if scores[i] < self.threshold or not any(len(p) >= 3 for p in c["polys"]):
                entry["status"] = "below_threshold" if scores[i] < self.threshold else "empty_mask"
                continue
            if sel["max_bottles"] and len(bottles) >= sel["max_bottles"]:
                entry["status"] = "over_max_bottles"
                continue
            crop, info = render_bottle(rgb, c["polys"], cfg)
            if sel["min_bottle_px"] and info["bottle_size_px"][1] < sel["min_bottle_px"]:
                entry["status"] = "too_small"
                continue
            entry["status"] = "selected"
            n = len(bottles) + 1
            fname = f"{path.stem}_b{n}.{cfg['output']['format']}"
            params = {"jpg": [cv2.IMWRITE_JPEG_QUALITY, cfg["output"]["quality"]],
                      "webp": [cv2.IMWRITE_WEBP_QUALITY, cfg["output"]["quality"]], "png": []}[cfg["output"]["format"]]
            cv2.imwrite(str(out_dir / fname), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), params)
            bottles.append({"index": n, "candidate": c["id"], "score": entry["score"], "conf": c["conf"],
                            "box": c["box"], "output": fname, **info})
        timing["render"] = time.perf_counter() - t
        timing["total"] = time.perf_counter() - t0

        result = {
            "version": 1,
            "status": "ok" if bottles else "no_bottles",
            "source": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "width": seg["w"], "height": seg["h"]},
            "segmentation": {"model": seg["model"], "prompt": seg["prompt"], "imgsz": seg["imgsz"], "scale": seg["scale"],
                             "n_candidates": len(seg["cands"])},
            "selection": {"calibration": self.cfg["selection"]["calibration"], "threshold": round(self.threshold, 4),
                          "candidates": cands},
            "bottles": bottles,
            "config": cfg,
            "timing_ms": {k: round(v * 1000) for k, v in timing.items()},
        }
        if cfg["output"]["save_json"]:
            (out_dir / f"{path.stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
        return result


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
        A = np.array(info["matrix_src_to_dst"]); Ainv = np.array(info["matrix_dst_to_src"])
        neck_dst = affine(A, np.array([info["neck_point"]]))[0]
        base_dst = affine(A, np.array([info["base_point"]]))[0]
        assert neck_dst[1] < base_dst[1], (deg, neck_dst, base_dst)                   # горлышко выше дна
        assert abs(neck_dst[0] - base_dst[0]) < 1.5, (deg, neck_dst, base_dst)        # строго вертикально
        assert info["neck_method"] == "profile", (deg, info["neck_method"])
        back = affine(Ainv, affine(A, np.array([[10.0, 20.0]])))[0]
        assert np.allclose(back, [10, 20], atol=1e-3), back                            # матрицы взаимно обратны
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
    if not args.inputs:
        ap.error("нужна хотя бы одна картинка или папка")
    cfg = load_config(args.config, args.set)
    base = args.config.resolve().parent
    files = [f for p in args.inputs for f in (sorted(p.iterdir()) if p.is_dir() else [p]) if f.suffix.lower() in EXTS]
    out_dir = resolve(cfg["output"]["dir"], base)
    out_dir.mkdir(parents=True, exist_ok=True)
    norm = Normalizer(cfg, base)
    for f in files:
        try:
            r = norm(f, out_dir)
            print(f"{f.name}: {r['status']}, бутылок {len(r['bottles'])} из {r['segmentation']['n_candidates']} кандидатов, "
                  f"{r['timing_ms']['total']} мс")
        except Exception as e:  # одна битая картинка не должна ронять весь пакет
            print(f"{f.name}: ошибка {e!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
