"""Сегментация бутылок: SAM3 по текстовому промпту или YOLO-seg.

Формат результата:
  {"w", "h", "model", "prompt", "imgsz", "scale", "ms", "cands": [{"id", "conf", "box", "area", "polys", "label"?}]}
Координаты боксов и полигонов даны в пикселях оригинала после EXIF-поворота.
Если задан label_prompt, у кандидата есть "label": {"conf", "cover"} (см. label_stats).
"""
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

BOTTLE = 39  # класс bottle в COCO


def open_image(path) -> Image.Image:
    # EXIF-поворот применяем явно, иначе координаты разойдутся с тем, как картинку видит человек
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def label_stats(bottle: np.ndarray, label_confs, label_masks, min_conf: float) -> dict:
    """Видимость этикетки на бутылке по маскам SAM3.

    conf: лучшая уверенность среди этикеток, лежащих в основном внутри бутылки (>= 50% площади этикетки).
    cover: доля площади бутылки под такими этикетками с уверенностью >= min_conf.
    """
    ys, xs = np.nonzero(bottle)
    if len(xs) == 0:
        return {"conf": 0.0, "cover": 0.0}
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1  # считаем только в боксе бутылки
    b = bottle[y1:y2, x1:x2]
    covered = np.zeros_like(b)
    best = 0.0
    for c, lm in zip(label_confs, label_masks):
        crop = lm[y1:y2, x1:x2]
        inter = np.logical_and(crop, b).sum()
        if inter == 0 or inter < 0.5 * lm.sum():
            continue
        best = max(best, float(c))
        if c >= min_conf:
            covered |= crop
    return {"conf": round(best, 3), "cover": round(float(np.logical_and(covered, b).sum() / b.sum()), 3)}


class Segmenter:
    def __init__(self, model, prompt="wine bottle", conf=None, imgsz=None, max_side=1600,
                 label_prompt=None, label_conf=0.4):
        self.model = str(model)
        self.is_sam3 = "sam3" in Path(model).name
        if self.is_sam3 and not Path(model).is_file():
            raise FileNotFoundError(f"нет весов {model}, как скачать: см. README.md")
        self.prompt = prompt if self.is_sam3 else "class bottle"
        self.imgsz = imgsz or (1008 if self.is_sam3 else 640)
        self.conf = conf if conf is not None else (0.3 if self.is_sam3 else 0.1)  # SAM3 даёт бокалам 0.1–0.37, бутылкам 0.8+
        self.max_side = max_side  # маски 6000px на 20 кандидатов не влезут в VRAM
        self.label_prompt, self.label_conf = label_prompt, label_conf
        if label_prompt and not self.is_sam3:
            raise ValueError("проверка этикетки работает только с SAM3")
        if label_prompt and label_conf < self.conf:
            raise ValueError(f"label_conf {label_conf} ниже conf сегментации {self.conf}: такие этикетки SAM3 не вернёт")
        self._run = None

    @property
    def tag(self) -> str:
        """Идентификатор настроек сегментации; калибровка записывает, под какие настройки обучена."""
        return f"{Path(self.model).stem}_{self.imgsz}_c{self.conf}" + (f"_{self.prompt.replace(' ', '-')}" if self.is_sam3 else "")

    def _load(self):
        if self.is_sam3:
            from ultralytics.models.sam import SAM3SemanticPredictor
            p = SAM3SemanticPredictor(overrides=dict(model=self.model, conf=self.conf, imgsz=self.imgsz, quantize=16,
                                                     task="segment", mode="predict", save=False, verbose=False))

            def run(bgr):
                p.set_image(bgr)
                bottles = p(text=[self.prompt])[0]
                # этикетка отдельным запросом на тех же признаках: совместный промпт менял набор бутылок
                labels = p(text=[self.label_prompt])[0] if self.label_prompt else None
                return bottles, labels
            return run
        from ultralytics import YOLO
        m = YOLO(self.model)
        return lambda bgr: (m.predict(bgr, classes=[BOTTLE], conf=self.conf, imgsz=self.imgsz,
                                      retina_masks=True, verbose=False)[0], None)

    def __call__(self, img: Image.Image) -> dict:
        """Не потокобезопасно: вызывающий держит лок, если зовёт из нескольких потоков."""
        w, h = img.size
        k = min(1.0, self.max_side / max(w, h))
        small = img.resize((round(w * k), round(h * k)), Image.BILINEAR) if k < 1 else img
        bgr = np.ascontiguousarray(np.asarray(small)[:, :, ::-1])
        if self._run is None:
            self._run = self._load()
        t = time.perf_counter()
        r, lab = self._run(bgr)
        ms = round((time.perf_counter() - t) * 1000)
        if lab is not None and lab.masks is not None:
            label_confs, label_masks = lab.boxes.conf.cpu().numpy(), lab.masks.data.cpu().numpy() > 0.5
        else:
            label_confs, label_masks = [], []
        cands = []
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy() > 0.5  # маски уже в размере поданного кадра
            for m, box, conf in zip(masks, r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polys = [np.round(cv2.approxPolyDP(c, 1.0, True)[:, 0] / k).astype(int).tolist()
                         for c in contours if cv2.contourArea(c) > 10]
                cands.append({"conf": round(float(conf), 3), "box": [round(float(v) / k) for v in box],
                              "area": round(float(m.sum()) / m.size, 5), "polys": polys})
                if self.label_prompt:
                    cands[-1]["label"] = label_stats(m, label_confs, label_masks, self.label_conf)
        cands.sort(key=lambda c: -c["conf"])
        for i, c in enumerate(cands, 1):
            c["id"] = i
        return {"w": w, "h": h, "model": Path(self.model).name, "prompt": self.prompt, "imgsz": self.imgsz,
                "label_prompt": self.label_prompt, "scale": round(k, 4), "ms": ms, "cands": cands}
