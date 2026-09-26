import math
from dataclasses import dataclass

import cv2
import numpy as np

from vinishko.pipeline.steps.normalization.normalize import BORDERS
from vinishko.pipeline.steps.normalization.normalize import affine
from vinishko.pipeline.steps.normalization.normalize import all_points
from vinishko.pipeline.steps.normalization.normalize import rotated_extent
from vis_seacher_training.data.kernels import blend_color

MIN_WINDOW_PX = 16


@dataclass(frozen=True)
class ViewParams:
    """Как вырезать бутылку. Значения по умолчанию вместе с pad из normalize.toml повторяют render_bottle пайплайна."""

    pad_top: float
    pad_bottom: float
    """Запас окна над и под этикеткой в долях её высоты; окно не выходит за маску бутылки."""
    edge_jitter: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    """Сдвиг граней окна x1, y1, x2, y2 в долях его ширины и высоты; плюс — наружу. Шум окна маской не ограничен; аугментер боковые грани двигает только наружу."""
    angle_delta: float = 0.0
    """Добавка к углу выравнивания из разметки, градусы."""


@dataclass
class View:
    """Вырезка с оригинальным фоном и масками в её же координатах; фон заглушает отдельный шаг, после аугментаций."""

    image: np.ndarray
    bottle: np.ndarray
    label: np.ndarray
    inside: np.ndarray
    """Маски uint8 0/255: бутылка, этикетка и та часть вырезки, что лежит внутри исходного кадра."""


def polys_mask(polys: list, A: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Маска полигонов после аффинного преобразования A; size — ширина и высота."""
    mask = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillPoly(
        mask,
        [
            np.round(affine(A, np.asarray(p, np.float64))).astype(np.int32)
            for p in polys
            if len(p) >= 3
        ],
        255,
    )
    return mask


def mask_center(polys: list) -> np.ndarray:
    """Центр маски бутылки: тот же, что у bottle_axis пайплайна, той же растеризацией на ~400 px, но без оси и профиля ширины — они в 7 раз дороже, а здесь не нужны."""
    allp = all_points(polys)
    (x1, y1), (x2, y2) = allp.min(0), allp.max(0)
    s = 400 / max(x2 - x1, y2 - y1, 1)
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
    return ((np.stack([xs, ys], 1).astype(np.float64) - 1) / s + [x1, y1]).mean(0)


def render_view(rgb: np.ndarray, cand: dict, cfg: dict, params: ViewParams) -> View:
    """Поворот до вертикали и окно вокруг этикетки, как в render_bottle, но с управляемой геометрией и без заглушения фона.

    По ширине окно во всю маску бутылки, по высоте — этикетка с запасом, не дальше маски; затем грани сдвигает шум бокса.
    """
    polys, label = cand["bottle"], cand["label"]
    M, (rx1, ry1, bw, bh) = rotated_extent(
        polys, mask_center(polys), cand["angle"] + params.angle_delta
    )
    v = affine(M, all_points(label))
    ly1, ly2 = v[:, 1].min(), v[:, 1].max()
    label_h = ly2 - ly1
    wx1, wx2 = rx1, rx1 + bw
    wy1, wy2 = (
        max(ry1, ly1 - params.pad_top * label_h),
        min(ry1 + bh, ly2 + params.pad_bottom * label_h),
    )
    w, h = wx2 - wx1, wy2 - wy1
    jx1, jy1, jx2, jy2 = params.edge_jitter
    wx1, wy1, wx2, wy2 = wx1 - jx1 * w, wy1 - jy1 * h, wx2 + jx2 * w, wy2 + jy2 * h
    W, H = (
        max(MIN_WINDOW_PX, math.ceil(wx2 - wx1)),
        max(MIN_WINDOW_PX, math.ceil(wy2 - wy1)),
    )
    A = M.copy()
    A[:, 2] -= [wx1, wy1]
    image = cv2.warpAffine(
        rgb,
        A,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=BORDERS[cfg["crop"]["border"]],
        borderValue=(0, 0, 0),
    )
    inside = cv2.warpAffine(
        np.full(rgb.shape[:2], 255, np.uint8),
        A,
        (W, H),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return View(
        image, polys_mask(polys, A, (W, H)), polys_mask(label, A, (W, H)), inside
    )


def mute_background(
    image: np.ndarray, bottle: np.ndarray, cfg: dict, mask_scale: float = 1.0
) -> np.ndarray:
    """Заливает всё вне маски бутылки цветом background.color по растушёванной альфе, как render_bottle в режиме color.

    mask_scale растягивает расширение маски и ширину растушёвки: так в обучении имитируется неточная граница сегментации.
    """
    bg = cfg["background"]
    side = max(image.shape[:2])
    mask = bottle
    r = round(bg["dilate"] * mask_scale * side)
    if r > 0:
        mask = cv2.dilate(
            mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        )
    alpha = mask.astype(np.float32) / 255
    if bg["feather"] > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), bg["feather"] * mask_scale * side)
    return blend_color(
        np.ascontiguousarray(image), alpha, np.asarray(bg["color"], np.uint8)
    )
