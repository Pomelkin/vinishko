import json
import math
import re
import time
import tomllib
from pathlib import Path

import cv2
import numpy as np
import rich_click as click
from kostyl.utils import setup_logger
from PIL import Image

from vinishko.pipeline.steps.normalization.features import (
    candidate_features,
    gray_small,
)
from vinishko.pipeline.steps.normalization.seg import (
    ENV_DEVICE,
    Segmenter,
    label_stats,
    open_image,
)
from vinishko.pipeline.device import resolve_torch_device
from vinishko.pipeline.structs import BottleCrop, Reason, RejectedBottle
import torch

torch.set_float32_matmul_precision("high")

logger = setup_logger(fmt="detailed")

HERE = Path(__file__).resolve().parent
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BORDERS = {
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT_101,
    "constant": cv2.BORDER_CONSTANT,
}
# параметры, которых может не быть в конфиге, но их можно задать через --set
OPTIONAL = {"segmentation.model"}


# ---------- конфиг ----------


def load_config(path: Path, overrides: list[str]) -> dict:
    """Конфиг из TOML с переопределениями вида секция.ключ=значение."""
    cfg = tomllib.loads(path.read_text())
    for item in overrides:
        key, _, raw = item.partition("=")
        section, _, name = key.strip().partition(".")
        known = section in cfg and name in cfg[section]
        if not known and key.strip() not in OPTIONAL:
            raise SystemExit(f"--set {item}: нет параметра {key} в {path.name}")
        try:
            value = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw  # строку можно писать без кавычек
        cfg.setdefault(section, {})[name] = value
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


def neck_and_base(
    pts: np.ndarray, center: np.ndarray, axis: np.ndarray, orient: dict
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Концы оси: какой из них горлышко. Возвращает (горлышко, дно, способ, контраст профиля ширины)."""
    t = (pts - center) @ axis
    hist, edges = np.histogram(t, bins=20)
    widths = hist / max(edges[1] - edges[0], 1e-9)
    a_end, b_end = (
        widths[:3].mean(),
        widths[-3:].mean(),
    )  # ширина у концов «минус» и «плюс» оси
    contrast = float(abs(a_end - b_end) / (np.percentile(widths, 90) + 1e-9))
    p_minus, p_plus = center + axis * t.min(), center + axis * t.max()
    tilt = math.degrees(
        math.acos(min(1.0, abs(axis[1])))
    )  # отклонение оси от вертикали, 0..90
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
    cv2.fillPoly(
        m,
        [
            np.round((np.asarray(p, np.float64) - [x1, y1]) * s).astype(np.int32) + 1
            for p in polys
            if len(p) >= 3
        ],
        1,
    )
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


def polys_area(polys: list) -> float:
    """Суммарная площадь полигонов маски."""
    return float(sum(cv2.contourArea(np.asarray(p, np.float32)) for p in polys))


def main_parts(polys: list, min_frac: float) -> list:
    """Полигоны маски без мелких островков: остаются части не меньше min_frac площади самой крупной."""
    areas = [polys_area([p]) if len(p) >= 3 else 0.0 for p in polys]
    return [
        p
        for p, a in zip(polys, areas, strict=True)
        if a > 0 and a >= min_frac * max(areas)
    ]


def label_geometry(ax: dict, polys: list) -> dict:
    """Положение этикетки вдоль бутылки (доля полной длины от горлышка) и её размеры в системе бутылки."""
    lp = all_points(polys) - ax["neck"]
    full = max(
        ax["length"], 3.3 * ax["body_width"]
    )  # корпус может быть скрыт: полная длина не меньше 3.3 ширины
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


def rotated_extent(
    polys: list, center: np.ndarray, angle: float
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Матрица поворота вокруг центра маски и габариты маски после поворота: (M, (x1, y1, ширина, высота))."""
    M = cv2.getRotationMatrix2D(tuple(map(float, center)), angle, 1.0)
    verts = affine(M, all_points(polys))
    (rx1, ry1), (rx2, ry2) = verts.min(0), verts.max(0)
    return M, (rx1, ry1, rx2 - rx1, ry2 - ry1)


def label_window(
    M: np.ndarray, label: list, bottle: tuple[float, float, float, float], pad_y: float
) -> tuple[float, float, float, float]:
    """Окно (x1, y1, x2, y2) вокруг этикетки в повёрнутой системе координат бутылки.

    По ширине окно всегда во всю маску бутылки: разрыв между крупным планом и целой бутылкой есть только по высоте.
    По высоте — этикетка с запасом pad_y, доля её высоты, сверху и снизу, не дальше габаритов маски bottle, а значит и картинки:
    у крупного плана, где этикетка занимает почти всю видимую бутылку, окном остаётся вся маска.
    """
    v = affine(M, all_points(label))
    ly1, ly2 = v[:, 1].min(), v[:, 1].max()
    py = pad_y * (ly2 - ly1)
    bx1, by1, bx2, by2 = bottle
    return bx1, max(by1, ly1 - py), bx2, min(by2, ly2 + py)


def render_bottle(
    rgb: np.ndarray,
    polys: list,
    label: list,
    cfg: dict,
    angle: float | None = None,
    shift: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, dict]:
    """Кроп одной бутылки: поворот, окно вокруг этикетки label, заглушение фона по background.mode. Возвращает картинку и описание преобразования.

    Окно — полоса бутылки во всю её ширину на высоте этикетки с запасом crop.padding_y, не дальше маски бутылки, см. label_window:
    так целая бутылка с полки и крупный план этикетки дают кадры одного масштаба. Большой padding_y даёт всю бутылку.
    angle переопределяет угол, найденный по оси маски; shift сдвигает окно в долях его ширины и высоты.
    Оба нужны даталоадеру для джиттера поверх сохранённой разметки; запас и фон джиттерятся через cfg.
    """
    ax = bottle_axis(polys, cfg["orientation"])
    if angle is None:
        angle = ax["angle"]
    M, (rx1, ry1, bw, bh) = rotated_extent(polys, ax["center"], angle)
    wx1, wy1, wx2, wy2 = label_window(
        M, label, (rx1, ry1, rx1 + bw, ry1 + bh), cfg["crop"]["padding_y"]
    )
    W, H = max(1, math.ceil(wx2 - wx1)), max(1, math.ceil(wy2 - wy1))
    A = M.copy()
    A[:, 2] -= [wx1 + shift[0] * W, wy1 + shift[1] * H]
    border = BORDERS[cfg["crop"]["border"]]
    crop = cv2.warpAffine(
        rgb, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=border, borderValue=(0, 0, 0)
    )
    inside = cv2.warpAffine(
        np.ones(rgb.shape[:2], np.uint8),
        A,
        (W, H),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )

    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(
        mask,
        [
            np.round(affine(A, np.asarray(p, np.float64))).astype(np.int32)
            for p in polys
            if len(p) >= 3
        ],
        255,
    )
    side = max(W, H)
    bg = cfg["background"]
    if bg["mode"] != "none":
        r = round(bg["dilate"] * side)
        if r > 0:
            mask = cv2.dilate(
                mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)),
            )
        alpha = mask.astype(np.float32) / 255
        if bg["feather"] > 0:
            alpha = cv2.GaussianBlur(alpha, (0, 0), bg["feather"] * side)
        back = (
            fast_blur(crop, bg["blur_sigma"] * side)
            if bg["mode"] == "blur"
            else np.full_like(crop, bg["color"])
        )
        crop = (
            (alpha[..., None] * crop + (1 - alpha[..., None]) * back)
            .round()
            .astype(np.uint8)
        )

    out_h = cfg["output"]["height"]
    if out_h and out_h != H:
        k = out_h / H
        crop = cv2.resize(
            crop,
            (max(1, round(W * k)), out_h),
            interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC,
        )
        A = A * k
    info = {
        "angle_deg": round(angle, 2),
        "neck_method": ax["method"],
        "neck_profile_contrast": round(ax["contrast"], 3),
        "neck_point": [round(v, 1) for v in ax["neck"]],
        "base_point": [round(v, 1) for v in ax["base"]],
        "bottle_size_px": [round(bw, 1), round(bh, 1)],
        "window_size_px": [W, H],
        "out_of_bounds_frac": round(1 - float(inside.mean()), 4),
        "matrix_src_to_dst": np.round(A, 6).tolist(),
        "matrix_dst_to_src": np.round(cv2.invertAffineTransform(A), 6).tolist(),
        "width": crop.shape[1],
        "height": crop.shape[0],
    }
    return crop, info


def render_bottle_box(
    rgb: np.ndarray,
    polys: list,
    cfg: dict,
    angle: float | None = None,
    padding: tuple[float, float] | None = None,
) -> tuple[np.ndarray, dict]:
    """Вся бутылка как на обычном фото: поворот горлышком вверх, bbox повёрнутой маски с запасом, фон как есть.

    Это вход второго уровня и картинка позиции в каталоге: VLM, выбирающая ответ среди кандидатов, лучше работает по привычному фото,
    чем по окну этикетки с залитым фоном. Поворот и центр те же, что у render_bottle, поэтому оба кропа одной бутылки согласованы.
    padding — запас по x и y в долях ширины и высоты маски, по умолчанию bottle_crop.padding_x и padding_y конфига; за габариты кадра
    запас не выходит.
    """
    ax = bottle_axis(polys, cfg["orientation"])
    if angle is None:
        angle = ax["angle"]
    bc = cfg["bottle_crop"]
    pad_x, pad_y = padding if padding is not None else (bc["padding_x"], bc["padding_y"])
    M, (rx1, ry1, bw, bh) = rotated_extent(polys, ax["center"], angle)
    px, py = pad_x * bw, pad_y * bh
    h, w = rgb.shape[:2]
    frame = affine(M, np.array([[0, 0], [w, 0], [w, h], [0, h]], float))  # габариты самого кадра после поворота
    (fx1, fy1), (fx2, fy2) = frame.min(0), frame.max(0)
    # запас не выходит за кадр: у каталожных вырезок бутылка часто упирается в край, и replicate размножил бы его полосой
    wx1, wy1 = max(rx1 - px, fx1), max(ry1 - py, fy1)
    wx2, wy2 = min(rx1 + bw + px, fx2), min(ry1 + bh + py, fy2)
    W, H = max(1, math.ceil(wx2 - wx1)), max(1, math.ceil(wy2 - wy1))
    A = M.copy()
    A[:, 2] -= [wx1, wy1]
    crop = cv2.warpAffine(
        rgb, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=BORDERS[bc["border"]], borderValue=(0, 0, 0)
    )
    max_h = bc["max_height"]
    if max_h and max_h < H:
        k = max_h / H
        crop = cv2.resize(crop, (max(1, round(W * k)), max_h), interpolation=cv2.INTER_AREA)
        A = A * k
    info = {
        "angle_deg": round(angle, 2),
        "bottle_size_px": [round(float(bw), 1), round(float(bh), 1)],
        "padding_px": [round(float(px), 1), round(float(py), 1)],
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
    cv2.fillPoly(
        m, [np.asarray(p, np.int32) - [x1, y1] for p in polys if len(p) >= 3], 1
    )
    return m.astype(bool)


def render_label(
    rgb: np.ndarray, polys: list, angle: float, padding: float
) -> np.ndarray:
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
    return cv2.warpAffine(
        rgb, A, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def save_image(path: Path, rgb: np.ndarray, fmt: str, quality: int) -> None:
    """RGB-массив в файл; неудачная запись — ошибка, а не молчаливый пропуск."""
    params = {
        "jpg": [cv2.IMWRITE_JPEG_QUALITY, quality],
        "webp": [cv2.IMWRITE_WEBP_QUALITY, quality],
        "png": [],
    }[fmt]
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), params):
        raise OSError(f"не удалось записать {path}")


# ---------- прогон ----------


class NormalizationReason(Reason):
    """Почему нормализация не пропустила бутылку дальше: значение, заголовок для интерфейса, что случилось и каким параметром управляется."""

    NOT_TARGET = (
        "not_target",
        "Бутылка на фоне",
        "Бутылка не целевая: стоит на фоне, сбоку или обрезана; скор отбора ниже порога калибровки selection.threshold",
    )
    OVER_LIMIT = (
        "over_limit",
        "Лишняя бутылка",
        "Годных бутылок в кадре уже selection.max_bottles, у этой скор ниже",
    )
    BOTTLE_TOO_SMALL = (
        "bottle_too_small",
        "Бутылка слишком далеко",
        "Высота бутылки в кадре меньше selection.min_bottle_px",
    )
    NO_LABEL = (
        "no_label",
        "Этикетки не видно",
        "Бутылка повёрнута тылом или этикетки нет: этикетки закрывают меньше label.min_cover площади бутылки",
    )
    NOT_A_LABEL = (
        "not_a_label",
        "Это не основная этикетка",
        "Найдена только акцизная марка уже label_crop.min_width_frac ширины бутылки либо наклейка у горлышка выше label.min_position",
    )
    LABEL_PARTIAL = (
        "label_partial",
        "Этикетка видна частично",
        "Бутылка повёрнута боком или этикетка чем-то закрыта: занято меньше label.min_fill ширины бутылки",
    )
    LABEL_TOO_SMALL = (
        "label_too_small",
        "Этикетка слишком мелкая",
        "Короткая сторона этикетки меньше label.min_label_px пикселей, текст не прочитать",
    )
    LABEL_BLURRY = (
        "label_blurry",
        "Этикетка размыта",
        "Расфокус или смаз: резкость ниже label.min_sharpness",
    )


class Normalizer:
    """Сегментация → отбор по калибровке → проверка этикетки → кропы. Разметка без рендера — annotate, кропы с разметкой — вызов."""

    def __init__(
        self,
        cfg: dict | None = None,
        base: Path | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Читает калибровку и поднимает сегментатор.

        Без аргументов — normalize.toml рядом с модулем; устройство — segmentation.device конфига, переменная NORMALIZER_DEV его перекрывает,
        auto — cuda:0 при доступной CUDA, иначе cpu.
        base — папка, от которой считаются пути в cfg, по умолчанию директория модуля.
        """
        if cfg is None:
            cfg = load_config(HERE / "normalize.toml", [])
        self.cfg, self.base = cfg, base or HERE
        self.device = (
            device
            if device is not None
            else resolve_torch_device(ENV_DEVICE, cfg["segmentation"]["device"])
        )
        base = self.base
        sel, sc, lb = cfg["selection"], cfg["segmentation"], cfg["label"]
        self.calib = json.loads(resolve(sel["calibration"], base).read_text())
        self.threshold = (
            sel["threshold"] if sel["threshold"] >= 0 else self.calib["threshold"]
        )
        self.seg = Segmenter(
            resolve(sc["model"], base) if "model" in sc else None,
            sc["prompt"],
            sc["conf"],
            sc["imgsz"],
            sc["max_side"],
            label_prompt=lb["prompt"],
            label_conf=lb["min_conf"],
            exclude_prompts=lb["exclude_prompts"],
            device=self.device,
        )
        if self.seg.label_prompt is None:
            raise ValueError(
                "годная бутылка определяется по этикетке: нужны веса SAM3 и непустой label.prompt"
            )
        if self.seg.tag != self.calib.get("segmentation"):
            logger.warning(
                f"калибровка сделана для сегментации {self.calib.get('segmentation')}, сейчас {self.seg.tag}"
            )

    def largest_label(self, labels: list[dict], ax: dict) -> list | None:
        """Полигоны самой крупной этикетки корпуса; полоски уже min_width_frac ширины бутылки, то есть акцизные марки, не считаются.

        Маска одной этикетки у SAM3 бывает несвязной: к основному пятну прилипают островки на соседнем тексте и тиснении.
        Островки мельче min_part_frac площади основного пятна отбрасываются, иначе они растягивают бокс этикетки.
        """
        lc = self.cfg["label_crop"]
        solid = [
            main_parts(lab["polys"], lc["min_part_frac"])
            for lab in labels
            if lab["polys"]
        ]
        polys = [
            p
            for p in solid
            if p  # маска выродилась в точки или отрезки нулевой площади
            and label_geometry(ax, p)["width_frac"] >= lc["min_width_frac"]
        ]
        return max(polys, key=polys_area, default=None)

    def check_label(self, cand: dict, ax: dict, score: float) -> list | RejectedBottle:
        """Полигоны основной этикетки, если она годится, иначе отказ с причиной и числами.

        Порядок: есть ли этикетка, основная ли она, видна ли целиком, достаточно ли крупная и резкая.
        Положение этикетки вдоль бутылки проверяется только у бутылки, видимой целиком: у обрубка крупного плана
        неизвестно, где горлышко, и этикетка по центру кадра считалась бы наклейкой у горлышка.
        """
        lb, stats = self.cfg["label"], cand["label"]
        main = self.largest_label(cand["label_polys"], ax)
        if stats["cover"] < lb["min_cover"]:
            return RejectedBottle(
                NormalizationReason.NO_LABEL,
                f"этикетки закрывают {stats['cover']:.0%} площади бутылки, нужно от {lb['min_cover']:.0%}",
                score,
                cand["polys"],
                main,
            )
        if main is None:
            return RejectedBottle(
                NormalizationReason.NOT_A_LABEL,
                f"все наклейки уже {self.cfg['label_crop']['min_width_frac']:.0%} ширины бутылки",
                score,
                cand["polys"],
                None,
            )
        geo = label_geometry(ax, main)
        whole = ax["length"] >= lb["min_length_ratio"] * ax["body_width"]
        short_side = min(geo["along_px"], geo["across_px"])
        checks = [
            (
                NormalizationReason.NOT_A_LABEL,
                whole and geo["position"] < lb["min_position"],
                f"центр наклейки на {geo['position']:.0%} длины бутылки от горлышка, основная этикетка не выше {lb['min_position']:.0%}",
            ),
            (
                NormalizationReason.LABEL_PARTIAL,
                stats["fill"] < lb["min_fill"],
                f"этикетка занимает {stats['fill']:.0%} ширины бутылки, нужно от {lb['min_fill']:.0%}",
            ),
            (
                NormalizationReason.LABEL_TOO_SMALL,
                short_side < lb["min_label_px"],
                f"короткая сторона этикетки {short_side:.0f} px, нужно от {lb['min_label_px']} px",
            ),
            (
                NormalizationReason.LABEL_BLURRY,
                stats["sharpness"] < lb["min_sharpness"],
                f"резкость {stats['sharpness']:.0f}, нужно от {lb['min_sharpness']}",
            ),
        ]
        failed = next(((r, d) for r, bad, d in checks if bad), None)
        return (
            RejectedBottle(failed[0], failed[1], score, cand["polys"], main)
            if failed
            else main
        )

    def refine_label(
        self, img: Image.Image, cand: dict, ax: dict, scale: float, label: list
    ) -> list:
        """Если в первом проходе бутылка была мелкой, маска этикетки уточняется вторым проходом SAM3 по вырезке бутылки.

        Если второй проход этикетку не нашёл, остаётся маска первого: бутылка уже прошла проверку.
        """
        lc = self.cfg["label_crop"]
        side = np.ptp(all_points(cand["polys"]), axis=0).max() * (
            1 + 2 * lc["refine_margin"]
        )
        if min(1.0, self.seg.imgsz / max(side, 1)) / scale < lc["refine_min_gain"]:
            return label
        return (
            self.largest_label(
                self.seg.labels_on_crop(img, cand["polys"], lc["refine_margin"]), ax
            )
            or label
        )

    def judge(
        self,
        img: Image.Image,
        rgb: np.ndarray,
        cand: dict,
        score: float,
        scale: float,
        index: int,
    ) -> BottleCrop | RejectedBottle:
        """Целевая бутылка → годная с номером index и кропом или отказ: размер бутылки, затем этикетка, затем рендер."""
        ax = bottle_axis(cand["polys"], self.cfg["orientation"])
        # у бутылки короче min_aspect своих ширин ось PCA может лечь поперёк: такой кроп идёт без поворота
        axis_reliable = (
            ax["length"] >= self.cfg["fallback"]["min_aspect"] * ax["body_width"]
        )
        angle = ax["angle"] if axis_reliable else 0.0
        _, (_, _, _, bh) = rotated_extent(cand["polys"], ax["center"], angle)
        min_px = self.cfg["selection"]["min_bottle_px"]
        if bh < min_px:
            return RejectedBottle(
                NormalizationReason.BOTTLE_TOO_SMALL,
                f"высота бутылки {bh:.0f} px, нужно от {min_px} px",
                score,
                cand["polys"],
                None,
            )
        label = self.check_label(cand, ax, score)
        if isinstance(label, RejectedBottle):
            return label
        label = self.refine_label(img, cand, ax, scale, label)
        angle = round(angle, 2)
        crop, info = render_bottle(rgb, cand["polys"], label, self.cfg, angle=angle)
        box, box_info = render_bottle_box(rgb, cand["polys"], self.cfg, angle=angle)
        return BottleCrop(index, score, cand["polys"], label, angle, crop, info, box, box_info)

    def annotate_image(self, img: Image.Image) -> list[BottleCrop | RejectedBottle]:
        """Все бутылки, найденные SAM3, по убыванию скора отбора; каждая — BottleCrop с готовым кропом или RejectedBottle.

        Годных бутылок нет, если в списке нет ни одного BottleCrop. Координаты в пикселях img, то есть оригинала после EXIF-поворота.
        """
        rgb = np.asarray(img)
        seg = self.seg(img)
        scores = score_candidates(
            gray_small(img, self.calib["feat_side"]), seg, self.calib
        )
        max_bottles = self.cfg["selection"]["max_bottles"]
        items: list[BottleCrop | RejectedBottle] = []
        n_valid = 0
        for i in sorted(range(len(scores)), key=lambda i: -scores[i]):
            c = seg["cands"][i]
            if not any(len(p) >= 3 for p in c["polys"]):
                continue  # маска выродилась в точки: показывать и вырезать нечего
            score = round(scores[i], 4)
            if scores[i] < self.threshold:
                item = RejectedBottle(
                    NormalizationReason.NOT_TARGET,
                    f"скор отбора {score}, порог {self.threshold:.2f}",
                    score,
                    c["polys"],
                    None,
                )
            elif max_bottles and n_valid >= max_bottles:
                item = RejectedBottle(
                    NormalizationReason.OVER_LIMIT,
                    f"годных бутылок уже {max_bottles}",
                    score,
                    c["polys"],
                    None,
                )
            else:
                item = self.judge(img, rgb, c, score, seg["scale"], n_valid + 1)
            n_valid += isinstance(item, BottleCrop)
            items.append(item)
        if self.cfg["selection"]["bypass"] and not n_valid:
            w, h = (
                img.size
            )  # весь кадр как бутылка и этикетка, score 0: по нему дальше видно, что это обход, а отказы остаются в списке
            frame = [[[0, 0], [w, 0], [w, h], [0, h]]]
            crop, info = render_bottle(rgb, frame, frame, self.cfg, angle=0.0)
            box, box_info = render_bottle_box(rgb, frame, self.cfg, angle=0.0, padding=(0.0, 0.0))
            items.append(BottleCrop(1, 0.0, frame, frame, 0.0, crop, info, box, box_info))
        return items

    def annotate(self, path: Path | str) -> list[BottleCrop | RejectedBottle]:
        """Разметка одной картинки по пути; см. annotate_image."""
        return self.annotate_image(open_image(path))

    def __call__(
        self, img: Image.Image | Path | str
    ) -> list[BottleCrop | RejectedBottle]:
        """Картинка → все найденные бутылки: годные с кропами и отказы с причинами. Если BottleCrop в списке нет, дальше по пайплайну идти нечему."""
        if not isinstance(img, Image.Image):
            img = open_image(img)
        return self.annotate_image(img)


def remove_stale(out_dir: Path, stem: str) -> None:
    """Удаляет результаты прошлого прогона этой же картинки: бутылок могло стать меньше."""
    stale = re.compile(
        re.escape(stem) + r"_b\d+(\.(jpg|png|webp|json)|_label\.(npz|png)|_box\.(jpg|png|webp))"
    )
    for f in out_dir.iterdir():
        if stale.fullmatch(f.name):
            f.unlink()


def write_outputs(
    path: Path,
    out_dir: Path,
    rgb: np.ndarray,
    items: list[BottleCrop | RejectedBottle],
    cfg: dict,
) -> list[str]:
    """CLI: на каждую годную бутылку кроп поиска <имя>_bN.<формат>, вся бутылка <имя>_bN_box.<формат>, <имя>_bN.json, маска этикетки _label.npz и вырезка _label.png."""
    fmt, quality = cfg["output"]["format"], cfg["output"]["quality"]
    remove_stale(out_dir, path.stem)
    names = []
    for cand in items:
        if not isinstance(cand, BottleCrop):
            continue
        stem = f"{path.stem}_b{cand.index}"
        label_box = polys_box(cand.label)
        save_image(out_dir / f"{stem}.{fmt}", cand.crop, fmt, quality)
        save_image(out_dir / f"{stem}_box.{fmt}", cand.box_crop, fmt, quality)
        np.savez_compressed(
            out_dir / f"{stem}_label.npz", mask=polys_mask(cand.label, label_box)
        )
        save_image(
            out_dir / f"{stem}_label.png",
            render_label(rgb, cand.label, cand.angle, cfg["label_crop"]["padding"]),
            "png",
            0,
        )
        meta = {
            "source": str(path),
            "uuid": cand.uuid,
            "bottle_box": polys_box(cand.bottle),
            "bottle_score": cand.score,
            "bottle_angle": cand.crop_info["angle_deg"],
            "bottle_crop": f"{stem}.{fmt}",
            "bottle_crop_matrix": cand.crop_info["matrix_src_to_dst"],
            "bottle_box_crop": f"{stem}_box.{fmt}",
            "bottle_box_matrix": cand.box_info["matrix_src_to_dst"],
            "label_box": label_box,
            "label_mask": f"{stem}_label.npz",
            "label_crop": f"{stem}_label.png",
        }
        (out_dir / f"{stem}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1)
        )
        names.append(f"{stem}.{fmt}")
    return names


BODY_LABEL = np.array(
    [(-20, 40), (20, 40), (20, 80), (-20, 80)], float
)  # этикетка 40x40 на корпусе синтетической бутылки selftest


def check(cond: bool, *context: object) -> None:
    """Проверка selftest: падает с контекстом вместо assert, который отключается флагом -O."""
    if not cond:
        raise RuntimeError(f"selftest failed: {context}")


def selftest() -> None:
    """Синтетическая бутылка под разными углами: горлышко должно оказаться сверху, окно вокруг этикетки нужного размера, матрицы обратимы."""
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
        label = (BODY_LABEL @ R.T + [300, 250]).round().astype(int).tolist()
        crop, info = render_bottle(rgb, [poly], [label], cfg)
        A = np.array(info["matrix_src_to_dst"])
        Ainv = np.array(info["matrix_dst_to_src"])
        neck_dst = affine(A, np.array([info["neck_point"]]))[0]
        base_dst = affine(A, np.array([info["base_point"]]))[0]
        check(neck_dst[1] < base_dst[1], deg, neck_dst, base_dst)  # горлышко выше дна
        check(
            abs(neck_dst[0] - base_dst[0]) < 1.5, deg, neck_dst, base_dst
        )  # строго вертикально
        check(info["neck_method"] == "profile", deg, info["neck_method"])
        back = affine(Ainv, affine(A, np.array([[10.0, 20.0]])))[0]
        check(
            bool(np.allclose(back, [10, 20], atol=1e-3)), back
        )  # матрицы взаимно обратны
        bw, bh = info["bottle_size_px"]
        check(
            abs(crop.shape[0] - 40 * 1.4) <= 3 and abs(crop.shape[1] - bw) <= 2,
            deg,
            crop.shape,
            bw,
            bh,
        )  # этикетка 40x40: вся ширина бутылки, высота с запасом 20%
        check(
            115 <= bh <= 185 and 35 <= bw <= 45, deg, bw, bh
        )  # вертикальная бутылка 40x180
        box, binfo = render_bottle_box(rgb, [poly], cfg, angle=info["angle_deg"])
        px, py = cfg["bottle_crop"]["padding_x"] * bw, cfg["bottle_crop"]["padding_y"] * bh
        check(
            abs(box.shape[1] - (bw + 2 * px)) <= 2 and abs(box.shape[0] - (bh + 2 * py)) <= 2, deg, box.shape, bw, bh
        )  # вся бутылка с запасом, а не окно этикетки
        verts = affine(np.array(binfo["matrix_src_to_dst"]), np.asarray(poly, float))
        check(
            bool(verts.min() >= -1.5 and verts[:, 0].max() <= box.shape[1] + 1.5 and verts[:, 1].max() <= box.shape[0] + 1.5), deg, verts.min(0), verts.max(0)
        )  # маска целиком внутри кропа
    cfg["orientation"]["upright_within_deg"] = 30.0
    for deg, method in (
        (10, "upright_prior"),
        (170, "upright_prior"),
        (60, "profile"),
        (135, "profile"),
    ):
        r = math.radians(deg)
        R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
        poly = (outline @ R.T + [300, 250]).round().astype(int).tolist()
        label = (BODY_LABEL @ R.T + [300, 250]).round().astype(int).tolist()
        _, info = render_bottle(rgb, [poly], [label], cfg)
        check(info["neck_method"] == method, deg, info["neck_method"])
        if (
            deg == 170
        ):  # почти вертикальная бутылка вверх ногами считается стоящей: верхний конец становится горлышком
            check(abs(info["angle_deg"]) < 15, info["angle_deg"])
    selftest_labels(cfg, rgb, outline)
    print("selftest ok")


def selftest_labels(cfg: dict, rgb: np.ndarray, outline: np.ndarray) -> None:
    """Система координат бутылки, вырезка и маска этикетки, метрики видимости этикетки."""
    r = math.radians(30)
    R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])

    def to_img(pts: list | np.ndarray) -> list:
        return (np.array(pts, float) @ R.T + [300, 250]).round().astype(int).tolist()

    body = [(-20, 40), (20, 40), (20, 80), (-20, 80)]
    ax = bottle_axis(
        [to_img(outline)], cfg["orientation"]
    )  # бутылка 40x180: горлышко y -60..-5, корпус до 120
    check(
        abs(ax["length"] - 180) < 6 and abs(ax["body_width"] - 40) < 4,
        ax["length"],
        ax["body_width"],
    )
    body_label = label_geometry(ax, [to_img(body)])
    neck_label = label_geometry(
        ax, [to_img([(-7, -50), (7, -50), (7, -30), (-7, -30)])]
    )
    check(
        body_label["position"] > 0.5 and neck_label["position"] < 0.3,
        body_label,
        neck_label,
    )
    check(
        abs(body_label["along_px"] - 40) < 3
        and abs(body_label["width_frac"] - 1) < 0.15,
        body_label,
    )
    stub = label_geometry(
        {**ax, "length": 60.0}, [to_img(body)]
    )  # видно мало: длина берётся как 3.3 ширины
    check(
        abs(
            stub["position"]
            - body_label["position"] * ax["length"] / (3.3 * ax["body_width"])
        )
        < 0.01,
        stub,
    )

    # окно вокруг этикетки 40x40 на корпусе шириной 40: по ширине вся бутылка, по высоте 40 * 1.4
    window, _ = render_bottle(rgb, [to_img(outline)], [to_img(body)], cfg)
    check(
        abs(window.shape[1] - 40) <= 3 and abs(window.shape[0] - 56) <= 3, window.shape
    )
    low = [
        (-20, 90),
        (20, 90),
        (20, 118),
        (-20, 118),
    ]  # этикетка у самого дна: снизу окно обрезано маской бутылки
    window, _ = render_bottle(rgb, [to_img(outline)], [to_img(low)], cfg)
    check(abs(window.shape[0] - (28 * 1.2 + 2)) <= 3, window.shape)

    narrow = [
        (-5, 40),
        (5, 40),
        (5, 80),
        (-5, 80),
    ]  # узкая этикетка: окно всё равно во всю ширину бутылки
    window, _ = render_bottle(rgb, [to_img(outline)], [to_img(narrow)], cfg)
    check(
        abs(window.shape[1] - 40) <= 3 and abs(window.shape[0] - 56) <= 3, window.shape
    )

    lcrop = render_label(
        rgb, [to_img([(-30, 40), (30, 40), (30, 60), (-30, 60)])], ax["angle"], 0.03
    )
    check(
        abs(lcrop.shape[1] - 60 * 1.06) <= 3 and abs(lcrop.shape[0] - 20 * 1.06) <= 3,
        lcrop.shape,
    )  # снова горизонтальна
    square = [[[10, 20], [19, 20], [19, 29], [10, 29]]]
    box = polys_box(square)
    mask = polys_mask(square, box)
    check(
        box == [10, 20, 19, 29] and mask.shape == (10, 10) and bool(mask.all()),
        box,
        mask.shape,
    )

    bottle = np.zeros((100, 40), bool)
    bottle[10:90, 10:30] = True  # бутылка 20x80 = 1600 px
    front, outside, side, cap = (np.zeros_like(bottle) for _ in range(4))
    front[50:70, 10:30] = True  # этикетка 400 px внутри
    outside[0:20, 0:40] = True  # в основном снаружи бутылки
    side[50:70, 26:30] = True  # узкая полоска сбоку: повёрнута
    cap[10:20, 10:30] = True  # «этикетка» на крышке
    st = label_stats(bottle, [0.9, 0.95], [front, outside], 0.4)
    check(
        (st["conf"], st["cover"], st["fill"]) == (0.9, 0.25, 1.0), st
    )  # внешняя не считается, полоса закрыта
    check(
        label_stats(bottle, [0.3], [front], 0.4)["cover"] == 0.0
    )  # слабая не покрывает
    check(label_stats(bottle, [], [], 0.4)["cover"] == 0.0)
    check(label_stats(bottle, [0.9], [side], 0.4)["fill"] == 0.2)
    check(label_stats(bottle, [0.9], [cap], 0.4, exclude=cap)["cover"] == 0.0)
    big = np.zeros((1000, 400), bool)
    big[100:900, 100:300] = True  # резкость на крупной бутылке 200x800
    big_label = np.zeros_like(big)
    big_label[450:650, 100:300] = True
    noisy = np.full((1000, 400), 128, np.uint8)
    noisy[450:650, 100:300] = np.random.default_rng(1).integers(0, 255, (200, 200))
    check(
        label_stats(big, [0.9], [big_label], 0.4, gray_full=noisy)["sharpness"] > 300
    )  # мелкий контрастный рисунок
    check(
        label_stats(
            big, [0.9], [big_label], 0.4, gray_full=cv2.GaussianBlur(noisy, (0, 0), 6)
        )["sharpness"]
        < 300
    )  # расфокус
    check(
        label_stats(bottle, [0.9], [front], 0.4, gray_full=noisy[:100, :40])[
            "sharpness"
        ]
        < 300
    )  # 20 px: не читается


@click.command()
@click.argument("inputs", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=HERE / "normalize.toml",
    show_default=True,
)
@click.option(
    "--set",
    "overrides",
    multiple=True,
    metavar="секция.ключ=значение",
    help="Переопределить параметр конфига",
)
@click.option(
    "--selftest",
    "selftest_",
    is_flag=True,
    help="Проверка геометрии на синтетике, без модели",
)
def main(
    inputs: tuple[Path, ...], config: Path, overrides: tuple[str, ...], selftest_: bool
) -> None:
    """Нормализация бутылок вина: SAM3 → отбор по калибровке → поворот горлышком вверх → кроп с запасом → заливка фона.

    INPUTS — картинки или папки. На каждую бутылку в output.dir пишутся <имя>_bN.<формат> и <имя>_bN.json,
    <имя>_bN_label.npz с маской этикетки и <имя>_bN_label.png с вырезкой под OCR. Бутылки без годной этикетки не вырезаются.
    """
    if selftest_:
        selftest()
        return
    if not inputs:
        raise click.UsageError("нужна хотя бы одна картинка или папка")
    cfg = load_config(config, list(overrides))
    base = config.resolve().parent
    files = [
        f
        for p in inputs
        for f in (sorted(p.iterdir()) if p.is_dir() else [p])
        if f.suffix.lower() in EXTS
    ]
    out_dir = resolve(cfg["output"]["dir"], base)
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = [f.stem for f in files]
    if len(set(stems)) < len(
        stems
    ):  # выходные файлы называются по имени без расширения
        raise click.UsageError(
            f"одинаковые имена без расширения: {sorted({s for s in stems if stems.count(s) > 1})[:5]}"
        )
    if any(f.resolve().parent == out_dir.resolve() for f in files):
        raise click.UsageError(
            "output.dir совпадает с папкой исходников: очистка старых результатов могла бы удалить исходные картинки"
        )
    norm = Normalizer(cfg, base)
    for f in files:
        try:
            t0 = time.perf_counter()
            img = open_image(f)
            items = norm.annotate_image(img)
            names = write_outputs(f, out_dir, np.asarray(img), items, cfg)
            rejected = ", ".join(
                item.reason for item in items if isinstance(item, RejectedBottle)
            )
            click.echo(
                f"{f.name}: бутылок {len(items)}, кропов {len(names)}"
                f"{f', отказы: {rejected}' if rejected else ''}, {round((time.perf_counter() - t0) * 1000)} мс"
            )
        except Exception as e:  # одна битая картинка не должна ронять весь пакет
            logger.error(f"{f.name}: ошибка {e!r}")


if __name__ == "__main__":
    main()
