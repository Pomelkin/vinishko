"""Попиксельные ядра аугментаций на numba: смешивания и карты яркости, которые в numpy шли в несколько проходов с временными массивами.

Каждое ядро повторяет арифметику numpy-выражения, которое заменило, вплоть до типов промежуточных значений и округления half-to-even,
поэтому результат совпадает с прежним бит в бит. Без prange: воркеры даталоадера и так по одному на ядро процессора.
"""

import math

import numpy as np
from numba import njit

F32 = np.float32


@njit(cache=True)
def blend_color(image: np.ndarray, alpha: np.ndarray, color: np.ndarray) -> np.ndarray:
    """round(alpha·image + (1 − alpha)·color) в float32: заливка фона по растушёванной маске."""
    out = np.empty_like(image)
    for y in range(alpha.shape[0]):
        for x in range(alpha.shape[1]):
            a = alpha[y, x]
            b = F32(1.0) - a
            for c in range(3):
                out[y, x, c] = np.uint8(
                    round(a * F32(image[y, x, c]) + b * F32(color[c]))
                )
    return out


@njit(cache=True)
def blend_images(top: np.ndarray, bottom: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """round(alpha·top + (1 − alpha)·bottom) в float32: вмешивание порченой копии по маске этикетки."""
    out = np.empty_like(bottom)
    for y in range(alpha.shape[0]):
        for x in range(alpha.shape[1]):
            a = alpha[y, x]
            b = F32(1.0) - a
            for c in range(3):
                out[y, x, c] = np.uint8(
                    round(a * F32(top[y, x, c]) + b * F32(bottom[y, x, c]))
                )
    return out


@njit(cache=True)
def screen_gauss(
    image: np.ndarray, cx: int, cy: int, sx: float, sy: float, intensity: float
) -> None:
    """image += blob·(255 − image), blob — гауссово пятно; image float32, пятно в float64, как считал numpy."""
    for y in range(image.shape[0]):
        dy = (y - cy) / sy
        for x in range(image.shape[1]):
            dx = (x - cx) / sx
            blob = math.exp(-(dx * dx + dy * dy) / 2) * intensity
            for c in range(3):
                v = image[y, x, c]
                image[y, x, c] = F32(np.float64(v) + blob * np.float64(F32(255.0) - v))


@njit(cache=True)
def screen_map(image: np.ndarray, blob: np.ndarray) -> None:
    """image += blob·(255 − image) в float32: блик по готовой карте."""
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            b = blob[y, x]
            for c in range(3):
                v = image[y, x, c]
                image[y, x, c] = v + b * (F32(255.0) - v)


@njit(cache=True)
def gain_gradient(
    height: int, width: int, cos_a: float, sin_a: float, low: float, high: float
) -> np.ndarray:
    """Коэффициент яркости от low до high вдоль направления (cos_a, sin_a); координаты нормируются в float32, дальше float64, как в numpy."""
    gain = np.empty((height, width), np.float64)
    corners = np.empty(4, np.float64)
    k = 0
    for y in (0, height - 1):
        for x in (0, width - 1):
            corners[k] = (
                np.float64(F32(x) / F32(width) - F32(0.5)) * cos_a
                + np.float64(F32(y) / F32(height) - F32(0.5)) * sin_a
            )
            k += 1
    lo, hi = corners.min(), corners.max()  # функция линейна, крайние значения в углах
    span = hi - lo
    for y in range(height):
        ay = np.float64(F32(y) / F32(height) - F32(0.5)) * sin_a
        for x in range(width):
            along = np.float64(F32(x) / F32(width) - F32(0.5)) * cos_a + ay
            gain[y, x] = low + (high - low) * (along - lo) / span
    return gain


@njit(cache=True)
def apply_gain(
    image: np.ndarray, gain: np.ndarray, lift: np.ndarray, denom: np.ndarray
) -> np.ndarray:
    """clip(round(image·gain + lift·(gain − 1)/denom)); lift и denom — 0-мерные массивы типа gain, чтобы арифметика шла в его точности."""
    out = np.empty_like(image)
    one = gain.dtype.type(1)
    for y in range(gain.shape[0]):
        for x in range(gain.shape[1]):
            g = gain[y, x]
            add = lift[()] * (g - one) / denom[()]
            for c in range(3):
                v = image[y, x, c] * g + add
                out[y, x, c] = np.uint8(round(min(max(v, 0), 255)))
    return out
