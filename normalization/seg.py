"""Сегментация бутылок SAM3 по текстовому промпту и видимость этикетки.

Формат результата Segmenter:
  {"w", "h", "scale", "cands": [{"id", "conf", "box", "area", "polys", "label"?, "label_polys"?}]}
Координаты боксов и полигонов даны в пикселях оригинала после EXIF-поворота.
Если задан label_prompt, у кандидата есть "label": {"conf", "cover", "fill", "sharpness"} (см. label_stats)
и "label_polys": [{"conf", "polys"}] — маски засчитанных этикеток корпуса.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def open_image(path) -> Image.Image:
    # EXIF-поворот применяем явно, иначе координаты разойдутся с тем, как картинку видит человек
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def band_fill(bottle: np.ndarray, label: np.ndarray) -> float:
    """Какую долю полосы бутылки на высоте этикетки она закрывает.

    Полоса берётся поперёк оси бутылки между 5-м и 95-м перцентилем положения этикетки вдоль оси.
    Этикетка, повёрнутая боком или закрытая рукой, закрывает малую долю полосы.
    """
    ys, xs = np.nonzero(bottle)
    ly, lx = np.nonzero(label)
    if len(lx) < 10 or len(xs) < 30:
        return 0.0
    step = max(1, len(xs) // 20000)  # ось по подвыборке точек, на больших масках так быстрее
    pts = np.stack([xs[::step], ys[::step]], 1).astype(np.float64)
    c = pts.mean(0)
    axis = np.linalg.eigh(np.cov((pts - c).T))[1][:, -1]
    t_b = (np.stack([xs, ys], 1) - c) @ axis
    t_l = (np.stack([lx, ly], 1) - c) @ axis
    lo, hi = np.percentile(t_l, 5), np.percentile(t_l, 95)
    return float(((t_l >= lo) & (t_l <= hi)).sum() / max(1, ((t_b >= lo) & (t_b <= hi)).sum()))


def label_sharpness(gray_full: np.ndarray, label: np.ndarray, offset, k: float) -> float:
    """Резкость этикетки: дисперсия лапласиана внутри её маски на вырезке, приведённой к высоте 160 px.

    Вырезка берётся из оригинала. Мелкую этикетку приведение растягивает, и она получает низкую резкость,
    как и размытая: для идентификации обе одинаково бесполезны. Край маски отрезается эрозией,
    иначе контраст этикетки с бутылкой дал бы резкость и размытой этикетке.
    """
    ys, xs = np.nonzero(label)
    if len(xs) < 20:
        return 0.0
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    oy, ox = offset
    crop = gray_full[int((y1 + oy) / k):int((y2 + oy) / k), int((x1 + ox) / k):int((x2 + ox) / k)].astype(np.float32)
    if crop.shape[0] < 2 or crop.shape[1] < 2:
        return 0.0
    H = 160
    s = H / crop.shape[0]
    size = (max(8, round(crop.shape[1] * s)), H)
    crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    m = cv2.resize(label[y1:y2, x1:x2].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    m = cv2.erode(m, np.ones((7, 7), np.uint8)) > 0
    return round(float(cv2.Laplacian(crop, cv2.CV_32F)[m].var()), 1) if m.sum() > 50 else 0.0


def body_labels(bottle: np.ndarray, label_confs, label_masks, min_conf: float, exclude: np.ndarray | None = None):
    """Этикетки корпуса бутылки в её боксе: >= 50% площади внутри бутылки и < 50% в зоне exclude
    (крышки и горлышки: логотип на крышке и кольцевая этикетка на горлышке не основная этикетка).

    Возвращает ((y1, x1) бокса, маска бутылки в боксе, [(conf, маска этикетки корпуса)], лучшая conf).
    В список попадают только этикетки с conf >= min_conf, маски уже без зоны exclude и обрезаны по бутылке.
    """
    ys, xs = np.nonzero(bottle)
    if len(xs) == 0:
        return (0, 0), bottle[:0, :0], [], 0.0
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    b = bottle[y1:y2, x1:x2]
    ex = exclude[y1:y2, x1:x2] if exclude is not None else np.zeros_like(b)
    kept, best = [], 0.0
    for c, lm in zip(label_confs, label_masks):
        crop = lm[y1:y2, x1:x2]
        inter = np.logical_and(crop, b).sum()
        if inter == 0 or inter < 0.5 * lm.sum() or np.logical_and(crop, ex).sum() >= 0.5 * crop.sum():
            continue
        best = max(best, float(c))
        if c >= min_conf:
            kept.append((float(c), crop & b & ~ex))
    return (y1, x1), b, kept, best


def stats_of(offset, b: np.ndarray, kept: list, best: float, gray_full: np.ndarray | None = None, k: float = 1.0) -> dict:
    """Метрики этикетки по результату body_labels, см. label_stats."""
    body = np.zeros_like(b)
    for _, m in kept:
        body |= m
    if not body.any():
        return {"conf": round(best, 3), "cover": 0.0, "fill": 0.0, "sharpness": 0.0}
    return {"conf": round(best, 3), "cover": round(float(body.sum() / b.sum()), 3), "fill": round(band_fill(b, body), 3),
            "sharpness": label_sharpness(gray_full, body, offset, k) if gray_full is not None else 0.0}


def label_stats(bottle: np.ndarray, label_confs, label_masks, min_conf: float,
                exclude: np.ndarray | None = None, gray_full: np.ndarray | None = None, k: float = 1.0) -> dict:
    """Видимость основной этикетки на бутылке по маскам SAM3 (все маски в масштабе k относительно оригинала).

    conf: лучшая уверенность среди этикеток корпуса. cover: доля площади бутылки под ними.
    fill: доля полосы бутылки на высоте этикетки (band_fill). sharpness: резкость (label_sharpness).
    """
    offset, b, kept, best = body_labels(bottle, label_confs, label_masks, min_conf, exclude)
    return stats_of(offset, b, kept, best, gray_full, k)


def mask_polys(mask: np.ndarray, offset, scale: float, origin=(0, 0)) -> list:
    """Полигоны маски в пикселях оригинала. offset задан как (y, x), origin как (x, y).

    Пиксель i уменьшенной маски покрывает в оригинале отрезок [i/scale, (i+1)/scale), поэтому точка переводится
    через центр пикселя: без этого правый и нижний края маски получались короче на 1/scale пикселей.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [np.round((cv2.approxPolyDP(c, 1.0, True)[:, 0] + [offset[1], offset[0]] + 0.5) / scale - 0.5 + origin).astype(int).tolist()
            for c in contours if cv2.contourArea(c) > 10]


def masks_of(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(уверенности, боксы xyxy, bool-маски) из результата ultralytics; пустые массивы, если ничего не найдено."""
    if result.masks is None:
        return np.zeros(0), np.zeros((0, 4)), np.zeros((0, 0, 0), bool)
    return (result.boxes.conf.cpu().numpy(), result.boxes.xyxy.cpu().numpy().astype(np.float64),
            result.masks.data.cpu().numpy() > 0.5)


class Segmenter:
    def __init__(self, model, prompt="wine bottle", conf=0.3, imgsz=1008, max_side=1600,
                 label_prompt=None, label_conf=0.4, exclude_prompts=()):
        if not Path(model).is_file():
            raise FileNotFoundError(f"нет весов {model}, как скачать: см. README.md")
        self.model, self.prompt = str(model), prompt
        self.conf = conf  # SAM3 даёт бокалам 0.1–0.37, бутылкам 0.8+
        self.imgsz = imgsz
        self.max_side = max_side  # маски 6000px на 20 кандидатов не влезут в VRAM
        self.label_prompt, self.label_conf, self.exclude_prompts = label_prompt, label_conf, list(exclude_prompts)
        if label_prompt and label_conf < conf:
            raise ValueError(f"label_conf {label_conf} ниже conf сегментации {conf}: такие этикетки SAM3 не вернёт")
        self.predictor = None

    @property
    def tag(self) -> str:
        """Идентификатор настроек сегментации; калибровка записывает, под какие настройки обучена."""
        return f"{Path(self.model).stem}_{self.imgsz}_c{self.conf}_{self.prompt.replace(' ', '-')}"

    def query(self, img: Image.Image, prompts) -> dict:
        """Кодирует кадр один раз и выполняет каждый промпт отдельным запросом на тех же признаках.

        Промпты не объединяются в один запрос: совместный промпт «бутылка + этикетка» менял набор бутылок.
        Не потокобезопасно: вызывающий держит лок, если зовёт из нескольких потоков.
        """
        if self.predictor is None:
            from ultralytics.models.sam import SAM3SemanticPredictor
            self.predictor = SAM3SemanticPredictor(overrides=dict(
                model=self.model, conf=self.conf, imgsz=self.imgsz, quantize=16,
                task="segment", mode="predict", save=False, verbose=False))
        self.predictor.set_image(np.ascontiguousarray(np.asarray(img)[:, :, ::-1]))
        return {p: masks_of(self.predictor(text=[p])[0]) for p in prompts}

    def _labels(self, results: dict):
        """(уверенности этикеток, их маски, зона крышек и горлышек или None) из результатов query."""
        lconf, _, lmasks = results[self.label_prompt]
        ex = [results[p][2] for p in self.exclude_prompts if len(results[p][2])]
        return lconf, lmasks, np.any(np.concatenate(ex), axis=0) if ex else None

    def __call__(self, img: Image.Image) -> dict:
        w, h = img.size
        k = min(1.0, self.max_side / max(w, h))
        small = img.resize((round(w * k), round(h * k)), Image.BILINEAR) if k < 1 else img
        results = self.query(small, [self.prompt] + ([self.label_prompt, *self.exclude_prompts] if self.label_prompt else []))
        if self.label_prompt:
            lconf, lmasks, exclude = self._labels(results)
            gray_full = np.asarray(img.convert("L"))
        cands = []
        for conf, box, m in zip(*results[self.prompt]):
            cand = {"conf": round(float(conf), 3), "box": [round(float(v) / k) for v in box],
                    "area": round(float(m.sum()) / m.size, 5), "polys": mask_polys(m, (0, 0), k)}
            if self.label_prompt:
                offset, b, kept, best = body_labels(m, lconf, lmasks, self.label_conf, exclude)
                cand["label"] = stats_of(offset, b, kept, best, gray_full, k)
                cand["label_polys"] = [{"conf": round(c, 3), "polys": mask_polys(lm, offset, k)} for c, lm in kept]
            cands.append(cand)
        cands.sort(key=lambda c: -c["conf"])
        for i, c in enumerate(cands, 1):
            c["id"] = i
        return {"w": w, "h": h, "scale": round(k, 4), "cands": cands}

    def labels_on_crop(self, img: Image.Image, bottle_polys: list, margin: float = 0.05) -> list:
        """Второй проход: SAM3 ищет этикетки на вырезке бутылки из оригинала, в более высоком разрешении.

        Маска бутылки берётся из первого прохода. Возвращает [{"conf", "polys"}] в пикселях оригинала.
        """
        pts = np.concatenate([np.asarray(p) for p in bottle_polys])
        (bx1, by1), (bx2, by2) = pts.min(0), pts.max(0)
        pad = margin * max(bx2 - bx1, by2 - by1)
        x1, y1 = int(max(0, bx1 - pad)), int(max(0, by1 - pad))
        x2, y2 = int(min(img.width, bx2 + pad + 1)), int(min(img.height, by2 + pad + 1))
        region = img.crop((x1, y1, x2, y2))
        s = min(1.0, self.imgsz / max(region.size))
        if s < 1:
            region = region.resize((max(1, round(region.width * s)), max(1, round(region.height * s))), Image.BILINEAR)
        lconf, lmasks, exclude = self._labels(self.query(region, [self.label_prompt, *self.exclude_prompts]))
        bottle = np.zeros((region.height, region.width), np.uint8)
        cv2.fillPoly(bottle, [np.round((np.asarray(p) - [x1, y1]) * s).astype(np.int32) for p in bottle_polys], 1)
        offset, _, kept, _ = body_labels(bottle.astype(bool), lconf, lmasks, self.label_conf, exclude)
        return [{"conf": round(c, 3), "polys": mask_polys(m, offset, s, (x1, y1))} for c, m in kept]
