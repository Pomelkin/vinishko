import hashlib
import json
import math
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
from vinishko.pipeline.steps.normalization.seg import Segmenter, open_image

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


def bottle_axis(polys: list, orient: dict) -> dict:
    """Ось бутылки по маске: центр, точки горлышка и дна, угол поворота до вертикали горлышком вверх."""
    arrs = [np.asarray(p, np.float32) for p in polys if len(p) >= 3]
    allp = np.concatenate(arrs)
    x1, y1 = allp.min(0)
    x2, y2 = allp.max(0)
    s = 256 / max(
        x2 - x1, y2 - y1, 1
    )  # профиль считаем на маске ~256 px, этого хватает
    m = np.zeros((int((y2 - y1) * s) + 3, int((x2 - x1) * s) + 3), np.uint8)
    cv2.fillPoly(
        m, [np.round((a - [x1, y1]) * s).astype(np.int32) + 1 for a in arrs], 1
    )
    ys, xs = np.nonzero(m)
    pts = (np.stack([xs, ys], 1).astype(np.float64) - 1) / s + [x1, y1]
    center = pts.mean(0)
    axis = np.linalg.eigh(np.cov((pts - center).T))[1][:, -1]
    t = (pts - center) @ axis
    hist, edges = np.histogram(t, bins=20)
    widths = hist / max(edges[1] - edges[0], 1e-9)
    wmax = np.percentile(widths, 90) + 1e-9
    a_end, b_end = (
        widths[:3].mean(),
        widths[-3:].mean(),
    )  # ширина у концов «минус» и «плюс» оси
    contrast = abs(a_end - b_end) / wmax
    p_minus, p_plus = center + axis * t.min(), center + axis * t.max()
    tilt = math.degrees(
        math.acos(min(1.0, abs(axis[1])))
    )  # отклонение оси от вертикали, 0..90
    upper = (p_minus, p_plus) if p_minus[1] < p_plus[1] else (p_plus, p_minus)
    if tilt <= orient["upright_within_deg"]:
        # профиль ширины путает дно с горлышком при дырах и бликах в маске; почти вертикальную бутылку считаем стоящей
        neck, base, method = *upper, "upright_prior"
    elif orient["neck"] == "profile" and contrast >= orient["min_profile_contrast"]:
        neck, base, method = (
            (p_minus, p_plus, "profile")
            if a_end < b_end
            else (p_plus, p_minus, "profile")
        )
    else:
        neck, base, method = *upper, "up"
    v = neck - base
    # cv2.getRotationMatrix2D(angle=a+90) переводит направление с углом a (ось y вниз) в (0, -1), то есть вверх
    angle = math.degrees(math.atan2(v[1], v[0])) + 90 if orient["enabled"] else 0.0
    return {
        "center": center,
        "neck": neck,
        "base": base,
        "angle": angle,
        "method": method,
        "contrast": float(contrast),
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
    verts = affine(M, np.concatenate([np.asarray(p, np.float64) for p in polys if len(p) >= 3]))
    (rx1, ry1), (rx2, ry2) = verts.min(0), verts.max(0)
    return M, (rx1, ry1, rx2 - rx1, ry2 - ry1)


def render_bottle(
    rgb: np.ndarray, polys: list, cfg: dict, angle: float | None = None, shift: tuple[float, float] = (0.0, 0.0)
) -> tuple[np.ndarray, dict]:
    """Кроп одной бутылки: поворот, запас, блюр фона. Возвращает картинку и описание преобразования.

    angle переопределяет угол, найденный по оси маски; shift сдвигает окно кропа в долях ширины и высоты бутылки.
    Оба нужны даталоадеру для джиттера поверх сохранённой разметки; запас и фон джиттерятся через cfg.
    """
    ax = bottle_axis(polys, cfg["orientation"])
    if angle is None:
        angle = ax["angle"]
    M, (rx1, ry1, bw, bh) = rotated_extent(polys, ax["center"], angle)
    px, py = cfg["crop"]["padding_x"] * bw, cfg["crop"]["padding_y"] * bh
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
        "out_of_bounds_frac": round(1 - float(inside.mean()), 4),
        "matrix_src_to_dst": np.round(A, 6).tolist(),
        "matrix_dst_to_src": np.round(cv2.invertAffineTransform(A), 6).tolist(),
        "width": crop.shape[1],
        "height": crop.shape[0],
    }
    return crop, info


# ---------- прогон ----------

SUMMARY_KEYS = ("candidate", "conf", "box", "score", "status")


@dataclass
class Crop:
    """Кроп одной бутылки: RGB-массив и как он получен из оригинала (матрицы в info)."""

    image: np.ndarray
    index: int
    score: float
    box: list[int]
    info: dict


class Normalizer:
    """Сегментация → отбор по калибровке → ось бутылки → кропы. Разметка без рендера — annotate, кропы — вызов."""

    def __init__(self, cfg: dict, base: Path) -> None:
        """Читает калибровку и поднимает сегментатор; base — папка, от которой считаются пути в cfg."""
        self.cfg, self.base = cfg, base
        sel = cfg["selection"]
        self.calib = json.loads(resolve(sel["calibration"], base).read_text())
        self.threshold = sel["threshold"] if sel["threshold"] >= 0 else self.calib["threshold"]
        sc = cfg["segmentation"]
        self.seg = Segmenter(resolve(sc["model"], base), sc["prompt"], sc["conf"], sc["imgsz"], sc["max_side"])
        if self.seg.tag != self.calib.get("segmentation"):
            print(
                f"внимание: калибровка сделана для сегментации {self.calib.get('segmentation')}, сейчас {self.seg.tag}",
                file=sys.stderr,
            )

    def annotate_image(self, img: Image.Image) -> dict:
        """Разметка без рендера: все кандидаты сегментации с масками и статусом отбора, у отобранных — ось бутылки.

        Полигоны в пикселях оригинала после EXIF-поворота. Кандидаты отсортированы по скору, отобранные нумеруются index с 1.
        Этого достаточно, чтобы даталоадер рендерил кроп через render_bottle с джиттером угла, окна, запаса и фона.
        """
        seg = self.seg(img)
        scores = score_candidates(gray_small(img, self.calib["feat_side"]), seg, self.calib)
        sel = self.cfg["selection"]
        cands, n_selected = [], 0
        for i in sorted(range(len(scores)), key=lambda i: -scores[i]):
            c = seg["cands"][i]
            entry = {"candidate": c["id"], "conf": c["conf"], "box": c["box"], "score": round(scores[i], 4), "polys": c["polys"]}
            cands.append(entry)
            if scores[i] < self.threshold:
                entry["status"] = "below_threshold"
                continue
            if not any(len(p) >= 3 for p in c["polys"]):
                entry["status"] = "empty_mask"
                continue
            if sel["max_bottles"] and n_selected >= sel["max_bottles"]:
                entry["status"] = "over_max_bottles"
                continue
            ax = bottle_axis(c["polys"], self.cfg["orientation"])
            _, (_, _, bw, bh) = rotated_extent(c["polys"], ax["center"], ax["angle"])
            if sel["min_bottle_px"] and bh < sel["min_bottle_px"]:
                entry["status"] = "too_small"
                continue
            n_selected += 1
            entry.update(
                status="selected",
                index=n_selected,
                angle_deg=round(ax["angle"], 2),
                neck_method=ax["method"],
                neck_profile_contrast=round(ax["contrast"], 3),
                neck_point=[round(v, 1) for v in ax["neck"]],
                base_point=[round(v, 1) for v in ax["base"]],
                bottle_size_px=[round(float(bw), 1), round(float(bh), 1)],
            )
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
        """Кропы отобранных бутылок по разметке, в порядке убывания скора."""
        crops = []
        for entry in ann["candidates"]:
            if entry["status"] != "selected":
                continue
            crop, info = render_bottle(rgb, entry["polys"], self.cfg)
            crops.append(Crop(crop, entry["index"], entry["score"], entry["box"], info))
        return crops

    def __call__(self, img: Image.Image | Path | str) -> list[Crop]:
        """Картинка → кропы бутылок. Пустой список, если ни одна бутылка не прошла отбор."""
        if not isinstance(img, Image.Image):
            img = open_image(img)
        return self.render(np.asarray(img), self.annotate_image(img))


def write_outputs(path: Path, out_dir: Path, ann: dict, crops: list[Crop], cfg: dict) -> list[str]:
    """CLI: кропы в файлы <имя>_b<N>.<формат> и, если включено, JSON с разметкой на картинку."""
    fmt = cfg["output"]["format"]
    params = {
        "jpg": [cv2.IMWRITE_JPEG_QUALITY, cfg["output"]["quality"]],
        "webp": [cv2.IMWRITE_WEBP_QUALITY, cfg["output"]["quality"]],
        "png": [],
    }[fmt]
    bottles, names = [], []
    for c in crops:
        fname = f"{path.stem}_b{c.index}.{fmt}"
        cv2.imwrite(str(out_dir / fname), cv2.cvtColor(c.image, cv2.COLOR_RGB2BGR), params)
        names.append(fname)
        bottles.append({"index": c.index, "score": c.score, "box": c.box, "output": fname, **c.info})
    if cfg["output"]["save_json"]:
        result = {
            "version": 1,
            "status": ann["status"],
            "source": {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "width": ann["width"],
                "height": ann["height"],
            },
            "segmentation": ann["segmentation"],
            "selection": {
                "calibration": cfg["selection"]["calibration"],
                "threshold": ann["threshold"],
                "candidates": [{k: e[k] for k in SUMMARY_KEYS} for e in ann["candidates"]],
            },
            "bottles": bottles,
            "config": cfg,
        }
        (out_dir / f"{path.stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
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
    print("selftest ok")


@click.command()
@click.argument("inputs", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("-c", "--config", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=HERE / "normalize.toml", show_default=True)
@click.option("--set", "overrides", multiple=True, metavar="секция.ключ=значение", help="Переопределить параметр конфига")
@click.option("--selftest", "selftest_", is_flag=True, help="Проверка геометрии на синтетике, без модели")
def main(inputs: tuple[Path, ...], config: Path, overrides: tuple[str, ...], selftest_: bool) -> None:
    """Нормализация бутылок вина: SAM3 → отбор по калибровке → поворот горлышком вверх → кроп с запасом → блюр фона.

    INPUTS — картинки или папки. На каждую картинку в output.dir пишутся <имя>_b<N>.<формат> и, если output.save_json, <имя>.json.
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
    norm = Normalizer(cfg, base)
    for f in files:
        try:
            t0 = time.perf_counter()
            img = open_image(f)
            ann = norm.annotate_image(img)
            crops = norm.render(np.asarray(img), ann)
            write_outputs(f, out_dir, ann, crops, cfg)
            click.echo(
                f"{f.name}: {ann['status']}, бутылок {len(crops)} из {ann['segmentation']['n_candidates']} кандидатов, "
                f"{round((time.perf_counter() - t0) * 1000)} мс"
            )
        except Exception as e:  # одна битая картинка не должна ронять весь пакет
            click.echo(f"{f.name}: ошибка {e!r}", err=True)


if __name__ == "__main__":
    main()
